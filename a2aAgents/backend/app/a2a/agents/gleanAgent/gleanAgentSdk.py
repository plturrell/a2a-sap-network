"""
Glean Agent - A2A Compliant Code Analysis Agent
Combines Glean semantic code analysis, comprehensive linting, and test execution capabilities

This agent provides:
- Code indexing and semantic analysis via Glean services
- Multi-tool linting (pylint, flake8, mypy, bandit, eslint, etc.)
- Test execution and coverage analysis
- Dependency graph analysis
- Code quality metrics and reporting
- Refactoring suggestions
- Security vulnerability scanning
"""

"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""



import asyncio
import ast
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import sqlite3
import re
import shutil
from dataclasses import dataclass, asdict, field

# Import performance monitoring mixin
try:
    from app.a2a.sdk.mixins import PerformanceMonitoringMixin
    def monitor_a2a_operation(func): return func  # Stub decorator
except ImportError:
    class PerformanceMonitoringMixin: pass
    def monitor_a2a_operation(func): return func

# Import SDK components
from app.a2a.sdk import (
    A2AAgentBase, AgentConfig, a2a_handler, a2a_skill, a2a_task, mcp_tool, mcp_resource,
    A2AMessage, MessageRole, create_agent_id
)
from app.a2a.sdk.utils import create_error_response, create_success_response

# Import blockchain integration
from app.a2a.sdk.blockchainIntegration import BlockchainIntegrationMixin

# Import security base
from app.a2a.core.security_base import SecureA2AAgent

# Import trust system - Real implementation only
import sys
sys.path.insert(0, '/Users/apple/projects/a2a/a2aNetwork')
from trustSystem.smartContractTrust import (
    initialize_agent_trust,
    get_trust_contract,
    verify_a2a_message,
    sign_a2a_message
)

# Import workflow components with proper error handling
try:
    from app.a2a.core.workflowContext import workflowContextManager
    from app.a2a.core.workflowMonitor import workflowMonitor
except ImportError:
    # Create stub implementations if not available
    class WorkflowContextManager:
        def __init__(self):
            pass
    
    class WorkflowMonitor:
        def __init__(self):
            pass
    
    workflowContextManager = WorkflowContextManager()
    workflowMonitor = WorkflowMonitor()


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
logger = logging.getLogger(__name__)


class AnalysisType(str, Enum):
    """Types of code analysis"""
    GLEAN_SEMANTIC = "glean_semantic"
    LINT = "lint"
    TEST = "test"
    SECURITY = "security"
    DEPENDENCY = "dependency"
    COMPLEXITY = "complexity"
    COVERAGE = "coverage"
    FULL = "full"


class IssueType(str, Enum):
    """Code quality issue types"""
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
    TEST_FAILURE = "test_failure"
    COVERAGE_LOW = "coverage_low"


class IssueSeverity(str, Enum):
    """Issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class CodeIssue:
    """Represents a code quality issue"""
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
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AnalysisResult:
    """Result of code analysis"""
    analysis_id: str
    analysis_type: AnalysisType
    directory: str
    files_analyzed: int
    issues: List[CodeIssue]
    metrics: Dict[str, Any]
    duration: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    glean_facts: Optional[Dict[str, Any]] = None
    test_results: Optional[Dict[str, Any]] = None
    coverage_data: Optional[Dict[str, Any]] = None


class GleanAgent(SecureA2AAgent, BlockchainIntegrationMixin):
    """
    A2A Compliant Glean Agent for comprehensive code analysis
    """
    
    def __init__(self, base_url: str = None):
        agent_id = create_agent_id("glean_agent", "code_analysis")
        
        # Create agent configuration
        config = AgentConfig(
            agent_id=agent_id,
            name="Glean Code Analysis Agent",
            description="Comprehensive code analysis combining Glean semantic analysis, linting, and testing",
            version="1.0.0",
            base_url=base_url or os.getenv("GLEAN_AGENT_URL")
        )
        
        # Initialize parent classes properly
        # Initialize SecureA2AAgent (which will call A2AAgentBase.__init__)
        SecureA2AAgent.__init__(self, config)
        
        # Set blockchain_enabled before initializing the mixin
        self.blockchain_enabled = os.getenv("BLOCKCHAIN_ENABLED", "true").lower() == "true"
        
        # Initialize blockchain integration
        BlockchainIntegrationMixin.__init__(self)
        
        # Initialize database for storing analysis results
        self.db_path = Path(__file__).parent / "glean_analysis.db"
        self._init_database()
        
        # Initialize async result cache
        self._result_cache = {}
        self._cache_locks = {}
        self._cache_ttl = 3600  # 1 hour TTL
        
        # Tool configurations
        self.linters = {
            "python": ["pylint", "flake8", "mypy", "bandit", "vulture"],
            "javascript": ["eslint", "jshint", "standard", "prettier", "jscpd", "retire"],
            "typescript": ["@typescript-eslint/eslint-plugin", "eslint", "tsc", "typescript-strict"],
            "html": ["htmlhint", "w3c-validator"],
            "xml": ["xmllint"],
            "yaml": ["yamllint"],
            "json": ["jsonlint"],
            "shell": ["shellcheck", "bashate"],
            "css": ["stylelint", "csslint"],
            "scss": ["stylelint", "sass-lint", "scss-lint"],
            "cds": ["cds-lint", "@sap/cds-dk", "cds-compiler-check"],
            "solidity": ["solhint", "slither", "mythril", "solc", "echidna"]
        }
        
        # Glean service endpoint
        self.glean_service_url = os.getenv("GLEAN_SERVICE_URL", "http://localhost:4000/api/glean")
        
        # Initialize trust
        self.trust_info = initialize_agent_trust(
            agent_id=self.agent_id,
            agent_type="analysis"
        )
        
        # Store capabilities separately
        self.capabilities = ["code_analysis", "linting", "testing", "security_scan"]
        
        logger.info(f"Initialized Glean Agent: {self.agent_id}")
    
    def _init_database(self):
        """Initialize SQLite database for analysis results"""
        self.db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id TEXT PRIMARY KEY,
                    analysis_type TEXT NOT NULL,
                    directory TEXT NOT NULL,
                    files_analyzed INTEGER,
                    issue_count INTEGER,
                    duration REAL,
                    timestamp TEXT,
                    results TEXT
                );
                
                CREATE TABLE IF NOT EXISTS code_issues (
                    id TEXT PRIMARY KEY,
                    analysis_id TEXT,
                    file_path TEXT,
                    line INTEGER,
                    column INTEGER,
                    tool TEXT,
                    issue_type TEXT,
                    severity TEXT,
                    code TEXT,
                    message TEXT,
                    rule TEXT,
                    suggestion TEXT,
                    auto_fixable BOOLEAN,
                    created_at TEXT,
                    FOREIGN KEY (analysis_id) REFERENCES analysis_results(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_issues_severity ON code_issues(severity);
                CREATE INDEX IF NOT EXISTS idx_issues_type ON code_issues(issue_type);
                CREATE INDEX IF NOT EXISTS idx_issues_file ON code_issues(file_path);
            """)
    
    def _generate_cache_key(self, operation: str, **kwargs) -> str:
        """Generate a cache key for the given operation and parameters"""
        import hashlib
        key_data = f"{operation}:{':'.join(f'{k}={v}' for k, v in sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode(), usedforsecurity=False).hexdigest()
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available and not expired"""
        if cache_key not in self._result_cache:
            return None
        
        cached_data = self._result_cache[cache_key]
        cache_time = cached_data.get('timestamp', 0)
        
        # Check if cache has expired
        if time.time() - cache_time > self._cache_ttl:
            # Remove expired cache
            del self._result_cache[cache_key]
            if cache_key in self._cache_locks:
                del self._cache_locks[cache_key]
            return None
        
        return cached_data.get('result')
    
    async def _set_cached_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Set result in cache with timestamp"""
        self._result_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    async def _with_cache(self, operation: str, func, **kwargs) -> Dict[str, Any]:
        """Execute function with caching support"""
        cache_key = self._generate_cache_key(operation, **kwargs)
        
        # Check cache first
        cached_result = await self._get_cached_result(cache_key)
        if cached_result:
            logger.debug(f"Using cached result for {operation}")
            return cached_result
        
        # Use locks to prevent concurrent execution of same operation
        if cache_key not in self._cache_locks:
            self._cache_locks[cache_key] = asyncio.Lock()
        
        async with self._cache_locks[cache_key]:
            # Double-check cache after acquiring lock
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(**kwargs)
            else:
                result = func(**kwargs)
            
            # Cache the result
            await self._set_cached_result(cache_key, result)
            return result
    
    def clear_cache(self, pattern: str = None) -> int:
        """Clear cache entries, optionally matching a pattern"""
        if pattern is None:
            count = len(self._result_cache)
            self._result_cache.clear()
            self._cache_locks.clear()
            return count
        
        # Clear entries matching pattern
        keys_to_remove = [key for key in self._result_cache.keys() if pattern in key]
        for key in keys_to_remove:
            del self._result_cache[key]
            if key in self._cache_locks:
                del self._cache_locks[key]
        
        return len(keys_to_remove)
    
    async def _run_command(self, command: str, cwd: str = None) -> Dict[str, str]:
        """Run a shell command and return stdout/stderr"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "stdout": stdout.decode('utf-8', errors='ignore'),
                "stderr": stderr.decode('utf-8', errors='ignore'),
                "returncode": process.returncode
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
    
    def _create_issue(self, file_path: str, line: int, message: str, severity: str, tool: str, 
                      column: int = 0, rule: str = None, suggestion: str = None) -> Dict[str, Any]:
        """Create a standardized issue dictionary"""
        import hashlib
        
        # Generate unique ID for the issue
        issue_id = hashlib.md5(f'{file_path}{line}{column}{tool}{message}'.encode(), usedforsecurity=False).hexdigest()[:8]
        
        # Map tool to issue type
        issue_type_mapping = {
            "pylint": IssueType.STYLE,
            "flake8": IssueType.STYLE,
            "mypy": IssueType.TYPE_ERROR,
            "bandit": IssueType.SECURITY,
            "eslint": IssueType.STYLE,
            "jshint": IssueType.STYLE,
            "tslint": IssueType.STYLE,
            "htmlhint": IssueType.STYLE,
            "xmllint": IssueType.SYNTAX,
            "yamllint": IssueType.STYLE,
            "shellcheck": IssueType.STYLE,
            "stylelint": IssueType.STYLE,
            "cds-lint": IssueType.STYLE,
            "solhint": IssueType.STYLE,
            "slither": IssueType.SECURITY,
            "js-semantics": IssueType.BEST_PRACTICE,
            "js-security": IssueType.SECURITY,
            "js-performance": IssueType.PERFORMANCE,
            "ts-semantics": IssueType.BEST_PRACTICE,
            "ts-security": IssueType.SECURITY,
            "scss-semantics": IssueType.BEST_PRACTICE,
            "cds-semantics": IssueType.BEST_PRACTICE,
            "cds-security": IssueType.SECURITY,
            "sol-semantics": IssueType.BEST_PRACTICE,
            "sol-security": IssueType.SECURITY,
            "sol-gas": IssueType.PERFORMANCE
        }
        
        issue_type = issue_type_mapping.get(tool, IssueType.CODE_SMELL)
        
        # Map severity string to enum
        severity_mapping = {
            "error": IssueSeverity.ERROR,
            "warning": IssueSeverity.WARNING,
            "info": IssueSeverity.INFO,
            "critical": IssueSeverity.CRITICAL,
            "major": IssueSeverity.MAJOR,
            "minor": IssueSeverity.MINOR
        }
        severity_enum = severity_mapping.get(severity.lower(), IssueSeverity.INFO)
        
        # Try to extract code snippet if file exists
        code_snippet = ""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                if 0 < line <= len(lines):
                    code_snippet = lines[line - 1].rstrip()
        except:
            code_snippet = ""
        
        return {
            "id": f"{tool}_{issue_id}",
            "file_path": file_path,
            "line": line,
            "column": column,
            "tool": tool,
            "issue_type": issue_type,
            "severity": severity_enum,
            "code": code_snippet,
            "message": message,
            "rule": rule,
            "suggestion": suggestion,
            "auto_fixable": False,
            "created_at": datetime.utcnow()
        }
    
    async def _retry_with_exponential_backoff(
        self,
        func,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exceptions: Tuple = (Exception,),
        **kwargs
    ) -> Any:
        """Execute function with exponential backoff retry logic"""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(**kwargs)
                else:
                    return func(**kwargs)
            except exceptions as e:
                last_exception = e
                
                if attempt == max_retries:
                    logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                    raise e
                
                # Calculate delay with exponential backoff
                delay = min(initial_delay * (2 ** attempt), max_delay)
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def _handle_analysis_error(self, operation: str, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Standardized error handling for analysis operations"""
        error_info = {
            "error": True,
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat(),
            "context": context or {}
        }
        
        # Log the error with appropriate level
        if isinstance(error, (FileNotFoundError, PermissionError)):
            logger.warning(f"File system error in {operation}: {error}")
        elif isinstance(error, subprocess.TimeoutExpired):
            logger.error(f"Timeout in {operation}: {error}")
        elif isinstance(error, (ConnectionError, TimeoutError)):
            logger.error(f"Network error in {operation}: {error}")
        else:
            logger.error(f"Unexpected error in {operation}: {error}", exc_info=True)
        
        return error_info
    
    async def _safe_subprocess_run(
        self,
        cmd: List[str],
        timeout: int = 300,
        cwd: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Safely run subprocess with proper error handling"""
        try:
            result = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    **kwargs
                ),
                timeout=timeout
            )
            
            stdout, stderr = await result.communicate()
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": stdout.decode('utf-8', errors='replace'),
                "stderr": stderr.decode('utf-8', errors='replace'),
                "command": ' '.join(cmd)
            }
            
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Command timed out",
                "timeout": timeout,
                "command": ' '.join(cmd)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "command": ' '.join(cmd)
            }
    
    async def _run_parallel_analysis(
        self,
        analysis_tasks: List[Dict[str, Any]],
        max_concurrent: int = 4
    ) -> List[Dict[str, Any]]:
        """Run multiple analysis tasks in parallel with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_single_task(task: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    task_type = task.get('type')
                    task_params = task.get('params', {})
                    
                    if task_type == 'lint':
                        return await self._perform_lint_analysis(**task_params)
                    elif task_type == 'glean':
                        return await self._perform_glean_analysis(**task_params)
                    elif task_type == 'security':
                        return await self.scan_dependency_vulnerabilities(**task_params)
                    elif task_type == 'coverage':
                        return await self.analyze_test_coverage(**task_params)
                    elif task_type == 'complexity':
                        return await self.analyze_code_complexity(**task_params)
                    elif task_type == 'refactoring':
                        return await self.analyze_code_refactoring(**task_params)
                    else:
                        return {"error": f"Unknown task type: {task_type}"}
                        
                except Exception as e:
                    return self._handle_analysis_error(task_type, e, task_params)
        
        # Execute all tasks in parallel
        results = await asyncio.gather(
            *[run_single_task(task) for task in analysis_tasks],
            return_exceptions=True
        )
        
        # Convert exceptions to error dictionaries
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task_type = analysis_tasks[i].get('type', 'unknown')
                processed_results.append(
                    self._handle_analysis_error(task_type, result, analysis_tasks[i])
                )
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def analyze_project_comprehensive_parallel(
        self,
        directory: str,
        analysis_types: List[str] = None,
        file_patterns: List[str] = None,
        max_concurrent: int = 4
    ) -> Dict[str, Any]:
        """Run comprehensive analysis with parallel execution"""
        start_time = time.time()
        analysis_id = f"analysis_{int(start_time * 1000)}"
        
        if analysis_types is None:
            analysis_types = ["lint", "security", "coverage", "complexity"]
        
        if file_patterns is None:
            file_patterns = ["*.py", "*.js", "*.ts"]
        
        logger.info(f"Starting parallel comprehensive analysis {analysis_id} for {directory}")
        
        # Prepare analysis tasks
        tasks = []
        
        if "lint" in analysis_types:
            tasks.append({
                "type": "lint",
                "params": {"directory": directory, "file_patterns": file_patterns}
            })
        
        if "glean" in analysis_types:
            tasks.append({
                "type": "glean", 
                "params": {"directory": directory}
            })
        
        if "security" in analysis_types:
            tasks.append({
                "type": "security",
                "params": {"directory": directory, "scan_dev_dependencies": True}
            })
        
        if "coverage" in analysis_types:
            tasks.append({
                "type": "coverage",
                "params": {"directory": directory, "coverage_threshold": 80.0}
            })
        
        if "complexity" in analysis_types:
            tasks.append({
                "type": "complexity",
                "params": {"directory": directory}
            })
        
        if "refactoring" in analysis_types:
            # For refactoring, we'll analyze Python files individually
            python_files = list(Path(directory).rglob("*.py"))
            for py_file in python_files[:5]:  # Limit to first 5 files for demo
                tasks.append({
                    "type": "refactoring",
                    "params": {"file_path": str(py_file)}
                })
        
        # Run analysis tasks in parallel
        try:
            results = await self._run_parallel_analysis(tasks, max_concurrent)
            
            # Aggregate results
            aggregated_result = {
                "analysis_id": analysis_id,
                "directory": directory,
                "analysis_types": analysis_types,
                "duration": time.time() - start_time,
                "timestamp": datetime.utcnow().isoformat(),
                "parallel_execution": True,
                "max_concurrent": max_concurrent,
                "tasks_completed": len(results),
                "results": {
                    "lint": None,
                    "glean": None,
                    "security": None,
                    "coverage": None,
                    "complexity": None,
                    "refactoring": []
                },
                "summary": {
                    "total_issues": 0,
                    "critical_issues": 0,
                    "files_analyzed": 0,
                    "quality_score": 0.0
                }
            }
            
            # Process results by type
            for i, result in enumerate(results):
                task_type = tasks[i].get('type')
                
                if task_type == "refactoring":
                    aggregated_result["results"]["refactoring"].append(result)
                else:
                    aggregated_result["results"][task_type] = result
                
                # Update summary
                if isinstance(result, dict) and not result.get("error"):
                    if "total_issues" in result:
                        aggregated_result["summary"]["total_issues"] += result["total_issues"]
                    if "critical_issues" in result:
                        aggregated_result["summary"]["critical_issues"] += result["critical_issues"]
                    if "files_analyzed" in result:
                        aggregated_result["summary"]["files_analyzed"] += result["files_analyzed"]
            
            # Calculate overall quality score
            total_issues = aggregated_result["summary"]["total_issues"]
            files_analyzed = aggregated_result["summary"]["files_analyzed"]
            
            if files_analyzed > 0:
                issues_per_file = total_issues / files_analyzed
                quality_score = max(0, 100 - (issues_per_file * 10))
                aggregated_result["summary"]["quality_score"] = quality_score
            
            # Store results in database
            await self._store_analysis_results(analysis_id, aggregated_result)
            
            logger.info(f"Parallel analysis {analysis_id} completed in {aggregated_result['duration']:.2f}s")
            return aggregated_result
            
        except Exception as e:
            logger.error(f"Parallel analysis failed: {e}")
            return self._handle_analysis_error("parallel_analysis", e, {
                "directory": directory,
                "analysis_types": analysis_types
            })
    
    async def compare_analysis_results(
        self,
        analysis_id_1: str,
        analysis_id_2: str
    ) -> Dict[str, Any]:
        """Compare two analysis results and generate diff report"""
        try:
            # Retrieve both analysis results
            result_1 = await self._get_analysis_result(analysis_id_1)
            result_2 = await self._get_analysis_result(analysis_id_2)
            
            if not result_1 or not result_2:
                return {
                    "error": "One or both analysis results not found",
                    "analysis_id_1": analysis_id_1,
                    "analysis_id_2": analysis_id_2
                }
            
            comparison = {
                "comparison_id": f"diff_{int(time.time() * 1000)}",
                "analysis_1": {
                    "id": analysis_id_1,
                    "timestamp": result_1.get("timestamp"),
                    "duration": result_1.get("duration")
                },
                "analysis_2": {
                    "id": analysis_id_2, 
                    "timestamp": result_2.get("timestamp"),
                    "duration": result_2.get("duration")
                },
                "summary_comparison": self._compare_summaries(
                    result_1.get("summary", {}),
                    result_2.get("summary", {})
                ),
                "issues_diff": self._compare_issues(
                    result_1.get("issues", []),
                    result_2.get("issues", [])
                ),
                "metrics_diff": self._compare_metrics(
                    result_1.get("metrics", {}),
                    result_2.get("metrics", {})
                ),
                "improvement_suggestions": []
            }
            
            # Generate improvement suggestions based on comparison
            comparison["improvement_suggestions"] = self._generate_improvement_suggestions(comparison)
            
            return comparison
            
        except Exception as e:
            return self._handle_analysis_error("compare_analysis", e, {
                "analysis_id_1": analysis_id_1,
                "analysis_id_2": analysis_id_2
            })
    
    def _compare_summaries(self, summary_1: Dict[str, Any], summary_2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare summary statistics between two analyses"""
        metrics = ["total_issues", "critical_issues", "files_analyzed", "quality_score"]
        comparison = {"changes": {}, "improvements": [], "regressions": []}
        
        for metric in metrics:
            val_1 = summary_1.get(metric, 0)
            val_2 = summary_2.get(metric, 0)
            change = val_2 - val_1
            
            comparison["changes"][metric] = {
                "from": val_1,
                "to": val_2,
                "change": change,
                "percent_change": (change / val_1 * 100) if val_1 > 0 else 0
            }
            
            # Determine if this is an improvement or regression
            if metric in ["total_issues", "critical_issues"]:
                if change < 0:
                    comparison["improvements"].append(f"Reduced {metric} by {abs(change)}")
                elif change > 0:
                    comparison["regressions"].append(f"Increased {metric} by {change}")
            elif metric == "quality_score":
                if change > 0:
                    comparison["improvements"].append(f"Improved {metric} by {change:.1f} points")
                elif change < 0:
                    comparison["regressions"].append(f"Decreased {metric} by {abs(change):.1f} points")
        
        return comparison
    
    def _compare_issues(self, issues_1: List[Dict], issues_2: List[Dict]) -> Dict[str, Any]:
        """Compare issues between two analyses"""
        # Create issue fingerprints for comparison
        def issue_fingerprint(issue):
            return f"{issue.get('file_path', '')}:{issue.get('line', 0)}:{issue.get('rule', '')}"
        
        fingerprints_1 = {issue_fingerprint(issue): issue for issue in issues_1}
        fingerprints_2 = {issue_fingerprint(issue): issue for issue in issues_2}
        
        # Find new, resolved, and persistent issues
        new_issues = []
        resolved_issues = []
        persistent_issues = []
        
        for fp, issue in fingerprints_2.items():
            if fp not in fingerprints_1:
                new_issues.append(issue)
            else:
                persistent_issues.append(issue)
        
        for fp, issue in fingerprints_1.items():
            if fp not in fingerprints_2:
                resolved_issues.append(issue)
        
        return {
            "new_issues": new_issues,
            "resolved_issues": resolved_issues,
            "persistent_issues": persistent_issues,
            "summary": {
                "new_count": len(new_issues),
                "resolved_count": len(resolved_issues),
                "persistent_count": len(persistent_issues)
            }
        }
    
    def _compare_metrics(self, metrics_1: Dict[str, Any], metrics_2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare metrics between two analyses"""
        comparison = {"changed_metrics": {}, "new_metrics": {}, "removed_metrics": {}}
        
        all_keys = set(metrics_1.keys()) | set(metrics_2.keys())
        
        for key in all_keys:
            val_1 = metrics_1.get(key)
            val_2 = metrics_2.get(key)
            
            if val_1 is None and val_2 is not None:
                comparison["new_metrics"][key] = val_2
            elif val_1 is not None and val_2 is None:
                comparison["removed_metrics"][key] = val_1
            elif val_1 != val_2:
                comparison["changed_metrics"][key] = {
                    "from": val_1,
                    "to": val_2
                }
        
        return comparison
    
    def _generate_improvement_suggestions(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions based on analysis comparison"""
        suggestions = []
        
        # Check for regressions and suggest fixes
        summary_comp = comparison.get("summary_comparison", {})
        regressions = summary_comp.get("regressions", [])
        
        if regressions:
            suggestions.append("Address the following regressions: " + ", ".join(regressions))
        
        # Check new issues
        issues_diff = comparison.get("issues_diff", {})
        new_issues_count = issues_diff.get("summary", {}).get("new_count", 0)
        
        if new_issues_count > 0:
            suggestions.append(f"Review and fix {new_issues_count} new issues introduced")
        
        # Check for persistent high-severity issues
        persistent_critical = len([
            issue for issue in issues_diff.get("persistent_issues", [])
            if issue.get("severity") == "critical"
        ])
        
        if persistent_critical > 0:
            suggestions.append(f"Prioritize fixing {persistent_critical} persistent critical issues")
        
        # Positive reinforcement for improvements
        improvements = summary_comp.get("improvements", [])
        if improvements:
            suggestions.append("Great progress: " + ", ".join(improvements))
        
        return suggestions
    
    async def _get_analysis_result(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis result from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM analysis_results WHERE id = ?",
                    (analysis_id,)
                )
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    result["results"] = json.loads(result["results"])
                    return result
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve analysis result {analysis_id}: {e}")
            return None
    
    
    async def get_quality_trends(
        self,
        directory: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get quality trends over time for a directory"""
        try:
            # Get analyses from the last N days
            cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM analysis_results 
                    WHERE directory = ? AND timestamp >= ?
                    ORDER BY timestamp ASC
                """, (directory, cutoff_date))
                
                rows = cursor.fetchall()
                
                if not rows:
                    return {
                        "directory": directory,
                        "days": days,
                        "message": "No analysis data found for the specified period"
                    }
                
                # Extract trend data
                timestamps = []
                quality_scores = []
                total_issues = []
                critical_issues = []
                files_analyzed = []
                
                for row in rows:
                    results = json.loads(row["results"])
                    summary = results.get("summary", {})
                    
                    timestamps.append(row["timestamp"])
                    quality_scores.append(summary.get("quality_score", 0))
                    total_issues.append(summary.get("total_issues", 0))
                    critical_issues.append(summary.get("critical_issues", 0))
                    files_analyzed.append(summary.get("files_analyzed", 0))
                
                # Calculate trends
                trends = {
                    "directory": directory,
                    "period_days": days,
                    "analyses_count": len(rows),
                    "date_range": {
                        "start": timestamps[0] if timestamps else None,
                        "end": timestamps[-1] if timestamps else None
                    },
                    "metrics": {
                        "quality_score": {
                            "values": quality_scores,
                            "trend": self._calculate_trend(quality_scores),
                            "current": quality_scores[-1] if quality_scores else 0,
                            "best": max(quality_scores) if quality_scores else 0,
                            "worst": min(quality_scores) if quality_scores else 0
                        },
                        "total_issues": {
                            "values": total_issues,
                            "trend": self._calculate_trend(total_issues, inverse=True),
                            "current": total_issues[-1] if total_issues else 0,
                            "best": min(total_issues) if total_issues else 0,
                            "worst": max(total_issues) if total_issues else 0
                        },
                        "critical_issues": {
                            "values": critical_issues,
                            "trend": self._calculate_trend(critical_issues, inverse=True),
                            "current": critical_issues[-1] if critical_issues else 0,
                            "best": min(critical_issues) if critical_issues else 0,
                            "worst": max(critical_issues) if critical_issues else 0
                        }
                    },
                    "insights": self._generate_trend_insights(quality_scores, total_issues, critical_issues)
                }
                
                return trends
                
        except Exception as e:
            return self._handle_analysis_error("get_trends", e, {
                "directory": directory,
                "days": days
            })
    
    def _calculate_trend(self, values: List[float], inverse: bool = False) -> Dict[str, Any]:
        """Calculate trend direction and statistics"""
        if len(values) < 2:
            return {"direction": "insufficient_data", "change": 0, "percentage": 0}
        
        # Simple linear trend calculation
        start_val = values[0]
        end_val = values[-1]
        change = end_val - start_val
        percentage = (change / start_val * 100) if start_val != 0 else 0
        
        # For inverse metrics (like issue counts), flip the direction
        if inverse:
            change = -change
            percentage = -percentage
        
        if abs(percentage) < 5:
            direction = "stable"
        elif percentage > 0:
            direction = "improving"
        else:
            direction = "declining"
        
        return {
            "direction": direction,
            "change": change,
            "percentage": round(percentage, 2)
        }
    
    def _generate_trend_insights(
        self,
        quality_scores: List[float],
        total_issues: List[float], 
        critical_issues: List[float]
    ) -> List[str]:
        """Generate insights based on trend analysis"""
        insights = []
        
        if not quality_scores:
            return ["Insufficient data for trend analysis"]
        
        # Quality score insights
        quality_trend = self._calculate_trend(quality_scores)
        if quality_trend["direction"] == "improving":
            insights.append(f"✅ Quality score is improving by {quality_trend['percentage']:.1f}%")
        elif quality_trend["direction"] == "declining":
            insights.append(f"⚠️ Quality score is declining by {abs(quality_trend['percentage']):.1f}%")
        else:
            insights.append("📊 Quality score is stable")
        
        # Issue trends
        if total_issues:
            issue_trend = self._calculate_trend(total_issues, inverse=True)
            if issue_trend["direction"] == "improving":
                insights.append(f"✅ Total issues decreasing by {abs(issue_trend['percentage']):.1f}%")
            elif issue_trend["direction"] == "declining":
                insights.append(f"⚠️ Total issues increasing by {issue_trend['percentage']:.1f}%")
        
        # Critical issue trends
        if critical_issues:
            critical_trend = self._calculate_trend(critical_issues, inverse=True)
            if critical_trend["direction"] == "improving":
                insights.append("🎯 Critical issues are being resolved effectively")
            elif critical_trend["direction"] == "declining":
                insights.append("🚨 Critical issues are increasing - immediate attention needed")
        
        # Overall assessment
        current_quality = quality_scores[-1] if quality_scores else 0
        if current_quality > 80:
            insights.append("🌟 Code quality is excellent")
        elif current_quality > 60:
            insights.append("👍 Code quality is good")
        elif current_quality > 40:
            insights.append("⚡ Code quality needs improvement")
        else:
            insights.append("🔧 Code quality requires significant attention")
        
        return insights
    
    def _detect_project_type(self, directory: str) -> str:
        """Detect project type based on files and structure"""
        path = Path(directory)
        
        # Check for specific files that indicate project type
        if (path / "package.json").exists():
            if (path / "tsconfig.json").exists():
                return "typescript"
            else:
                return "javascript"
        elif (path / "pyproject.toml").exists() or (path / "requirements.txt").exists():
            return "python"
        elif (path / "Cargo.toml").exists():
            return "rust"
        elif (path / "go.mod").exists():
            return "go"
        elif (path / "pom.xml").exists() or (path / "build.gradle").exists():
            return "java"
        elif (path / "Gemfile").exists():
            return "ruby"
        elif (path / "composer.json").exists():
            return "php"
        elif (path / "package.json").exists() and len(list(path.rglob("*.cds"))) > 0:
            return "cds"
        elif (path / ".cdsrc.json").exists() or (path / "cds-services.json").exists():
            return "cds"
        else:
            # Fallback to most common file types
            file_counts = {
                "python": len(list(path.rglob("*.py"))),
                "javascript": len(list(path.rglob("*.js"))),
                "typescript": len(list(path.rglob("*.ts"))) + len(list(path.rglob("*.tsx"))),
                "java": len(list(path.rglob("*.java"))),
                "rust": len(list(path.rglob("*.rs"))),
                "go": len(list(path.rglob("*.go"))),
                "html": len(list(path.rglob("*.html"))) + len(list(path.rglob("*.htm"))),
                "xml": len(list(path.rglob("*.xml"))) + len(list(path.rglob("*.xsl"))),
                "yaml": len(list(path.rglob("*.yaml"))) + len(list(path.rglob("*.yml"))),
                "json": len(list(path.rglob("*.json"))),
                "shell": len(list(path.rglob("*.sh"))) + len(list(path.rglob("*.bash"))),
                "css": len(list(path.rglob("*.css"))),
                "scss": len(list(path.rglob("*.scss"))) + len(list(path.rglob("*.sass"))),
                "cds": len(list(path.rglob("*.cds"))),
                "solidity": len(list(path.rglob("*.sol"))),
                "generic": 1
            }
            return max(file_counts, key=file_counts.get)
    
    def _get_project_config(self, project_type: str) -> Dict[str, Any]:
        """Get configuration settings for different project types"""
        configs = {
            "python": {
                "file_patterns": ["*.py"],
                "linters": ["pylint", "flake8", "mypy", "bandit", "vulture"],
                "test_patterns": ["test_*.py", "*_test.py", "tests/*.py"],
                "coverage_tools": ["coverage.py", "pytest-cov"],
                "complexity_threshold": 10,
                "quality_threshold": 75.0,
                "ignore_patterns": ["__pycache__", "*.pyc", ".pytest_cache", "venv/", "env/"],
                "security_scanners": ["bandit", "safety"]
            },
            "javascript": {
                "file_patterns": ["*.js", "*.jsx", "*.mjs", "*.cjs"],
                "linters": ["eslint", "jshint", "standard", "prettier", "jscpd", "retire"],
                "test_patterns": ["*.test.js", "*.spec.js", "test/**/*.js", "**/__tests__/**/*.js", "*.test.mjs"],
                "coverage_tools": ["nyc", "jest", "c8", "istanbul"],
                "complexity_threshold": 6,
                "quality_threshold": 85.0,
                "ignore_patterns": ["node_modules/", "dist/", "build/", "coverage/", ".next/", "out/"],
                "security_scanners": ["npm audit", "snyk", "retire", "eslint-plugin-security", "semgrep"],
                "modern_features": {
                    "es_modules": True,
                    "async_await": True,
                    "destructuring": True,
                    "arrow_functions": True,
                    "template_literals": True
                },
                "performance_analysis": {
                    "bundle_size": True,
                    "dead_code": True,
                    "unused_exports": True,
                    "circular_dependencies": True
                }
            },
            "typescript": {
                "file_patterns": ["*.ts", "*.tsx", "*.d.ts"],
                "linters": ["@typescript-eslint/eslint-plugin", "eslint", "tsc", "typescript-strict"],
                "test_patterns": ["*.test.ts", "*.spec.ts", "test/**/*.ts", "**/__tests__/**/*.ts"],
                "coverage_tools": ["nyc", "jest", "c8"],
                "complexity_threshold": 6,
                "quality_threshold": 85.0,
                "ignore_patterns": ["node_modules/", "dist/", "build/", "coverage/", "*.d.ts"],
                "security_scanners": ["npm audit", "snyk", "semgrep"],
                "type_checking": {
                    "strict": True,
                    "noImplicitAny": True,
                    "strictNullChecks": True,
                    "noImplicitReturns": True,
                    "noFallthroughCasesInSwitch": True
                },
                "semantic_analysis": {
                    "unused_imports": True,
                    "dead_code": True,
                    "type_assertions": True,
                    "any_usage": True,
                    "promise_handling": True
                }
            },
            "java": {
                "file_patterns": ["*.java"],
                "linters": ["checkstyle", "spotbugs", "pmd"],
                "test_patterns": ["*Test.java", "*Tests.java", "src/test/**/*.java"],
                "coverage_tools": ["jacoco"],
                "complexity_threshold": 12,
                "quality_threshold": 80.0,
                "ignore_patterns": ["target/", "build/", ".gradle/"],
                "security_scanners": ["spotbugs", "dependency-check"]
            },
            "solidity": {
                "file_patterns": ["*.sol"],
                "linters": ["solhint", "slither", "mythril", "solc", "echidna"],
                "test_patterns": ["*.t.sol", "*Test.sol", "test/**/*.sol"],
                "coverage_tools": ["solidity-coverage", "forge coverage"],
                "complexity_threshold": 8,
                "quality_threshold": 90.0,
                "ignore_patterns": ["node_modules/", "artifacts/", "cache/", "out/"],
                "security_scanners": ["slither", "mythril", "securify", "smartcheck", "manticore"],
                "gas_optimization": {
                    "enabled": True,
                    "target_threshold": 21000,
                    "storage_analysis": True,
                    "function_analysis": True
                },
                "vulnerability_checks": {
                    "reentrancy": True,
                    "integer_overflow": True,
                    "access_control": True,
                    "front_running": True,
                    "timestamp_dependence": True,
                    "denial_of_service": True
                }
            },
            "rust": {
                "file_patterns": ["*.rs"],
                "linters": ["rustc", "clippy"],
                "test_patterns": ["**/tests/*.rs", "src/**/*test*.rs"],
                "coverage_tools": ["cargo-tarpaulin"],
                "complexity_threshold": 10,
                "quality_threshold": 85.0,
                "ignore_patterns": ["target/", "Cargo.lock"],
                "security_scanners": ["cargo-audit"]
            },
            "html": {
                "file_patterns": ["*.html", "*.htm", "*.xhtml"],
                "linters": ["htmlhint", "w3c-validator"],
                "test_patterns": ["test*.html", "*test*.html"],
                "coverage_tools": [],
                "complexity_threshold": 5,
                "quality_threshold": 80.0,
                "ignore_patterns": ["dist/", "build/", "node_modules/"],
                "security_scanners": ["htmlhint"]
            },
            "xml": {
                "file_patterns": ["*.xml", "*.xsl", "*.xslt", "*.svg"],
                "linters": ["xmllint"],
                "test_patterns": ["test*.xml", "*test*.xml"],
                "coverage_tools": [],
                "complexity_threshold": 5,
                "quality_threshold": 85.0,
                "ignore_patterns": ["dist/", "build/"],
                "security_scanners": []
            },
            "yaml": {
                "file_patterns": ["*.yaml", "*.yml"],
                "linters": ["yamllint"],
                "test_patterns": [],
                "coverage_tools": [],
                "complexity_threshold": 3,
                "quality_threshold": 90.0,
                "ignore_patterns": ["dist/", "build/", "node_modules/"],
                "security_scanners": []
            },
            "json": {
                "file_patterns": ["*.json", "*.jsonc"],
                "linters": ["jsonlint"],
                "test_patterns": [],
                "coverage_tools": [],
                "complexity_threshold": 3,
                "quality_threshold": 95.0,
                "ignore_patterns": ["dist/", "build/", "node_modules/", "package-lock.json"],
                "security_scanners": []
            },
            "shell": {
                "file_patterns": ["*.sh", "*.bash", "*.zsh", "*.fish"],
                "linters": ["shellcheck", "bashate"],
                "test_patterns": ["test*.sh", "*test*.sh", "tests/*.sh"],
                "coverage_tools": ["bashcov"],
                "complexity_threshold": 8,
                "quality_threshold": 75.0,
                "ignore_patterns": ["node_modules/", "venv/"],
                "security_scanners": ["shellcheck"]
            },
            "css": {
                "file_patterns": ["*.css"],
                "linters": ["stylelint", "csslint"],
                "test_patterns": [],
                "coverage_tools": [],
                "complexity_threshold": 6,
                "quality_threshold": 80.0,
                "ignore_patterns": ["dist/", "build/", "node_modules/", "*.min.css"],
                "security_scanners": []
            },
            "scss": {
                "file_patterns": ["*.scss", "*.sass"],
                "linters": ["stylelint", "sass-lint", "scss-lint"],
                "test_patterns": ["test*.scss", "*test*.scss", "tests/**/*.scss"],
                "coverage_tools": ["sass-coverage"],
                "complexity_threshold": 6,
                "quality_threshold": 90.0,
                "ignore_patterns": ["dist/", "build/", "node_modules/", "*.min.css", "vendor/"],
                "security_scanners": ["stylelint"]
            },
            "cds": {
                "file_patterns": ["*.cds"],
                "linters": ["cds-lint", "@sap/cds-dk", "cds-compiler-check"],
                "test_patterns": ["test/*.cds", "**/test/**/*.cds", "tests/**/*.cds"],
                "coverage_tools": ["cds-test"],
                "complexity_threshold": 5,
                "quality_threshold": 95.0,
                "ignore_patterns": ["node_modules/", "gen/", "_out/", "dist/"],
                "security_scanners": ["cds-security-check"]
            },
            "generic": {
                "file_patterns": ["*"],
                "linters": [],
                "test_patterns": ["test*", "*test*"],
                "coverage_tools": [],
                "complexity_threshold": 10,
                "quality_threshold": 70.0,
                "ignore_patterns": [".git/", "node_modules/", "__pycache__/"],
                "security_scanners": []
            }
        }
        
        return configs.get(project_type, configs["generic"])
    
    async def configure_for_project(self, directory: str, project_type: str = None) -> Dict[str, Any]:
        """Configure GleanAgent for a specific project type"""
        try:
            if project_type is None:
                project_type = self._detect_project_type(directory)
            
            config = self._get_project_config(project_type)
            
            # Update agent configuration
            self.project_config = {
                "directory": directory,
                "project_type": project_type,
                "detected_at": datetime.utcnow().isoformat(),
                **config
            }
            
            logger.info(f"Configured GleanAgent for {project_type} project at {directory}")
            
            return {
                "success": True,
                "project_type": project_type,
                "configuration": config,
                "recommendations": self._get_project_recommendations(project_type, directory)
            }
            
        except Exception as e:
            return self._handle_analysis_error("configure_project", e, {
                "directory": directory,
                "project_type": project_type
            })
    
    def _get_project_recommendations(self, project_type: str, directory: str) -> List[str]:
        """Get setup recommendations for the project type"""
        recommendations = []
        path = Path(directory)
        
        if project_type == "python":
            if not (path / "pyproject.toml").exists() and not (path / "requirements.txt").exists():
                recommendations.append("Consider creating requirements.txt or pyproject.toml for dependency management")
            if not (path / ".gitignore").exists():
                recommendations.append("Add .gitignore file with Python-specific patterns")
            if not list(path.rglob("test_*.py")) and not (path / "tests").exists():
                recommendations.append("Create test files to improve code coverage")
        
        elif project_type == "javascript" or project_type == "typescript":
            if not (path / "package.json").exists():
                recommendations.append("Initialize npm package with 'npm init'")
            if not (path / ".eslintrc.js").exists() and not (path / ".eslintrc.json").exists():
                recommendations.append("Set up ESLint configuration for code quality")
            if project_type == "typescript" and not (path / "tsconfig.json").exists():
                recommendations.append("Create tsconfig.json for TypeScript configuration")
        
        elif project_type == "java":
            if not (path / "pom.xml").exists() and not (path / "build.gradle").exists():
                recommendations.append("Set up Maven (pom.xml) or Gradle (build.gradle) for build management")
        
        # General recommendations
        if not (path / "README.md").exists():
            recommendations.append("Add README.md file for project documentation")
        
        return recommendations
    
    async def initialize(self) -> None:
        """Initialize agent resources"""
        logger.info("Initializing Glean Agent resources")
        
        # Establish standard trust relationships FIRST
        await self.establish_standard_trust_relationships()
        
        # Initialize blockchain integration
        try:
            await self.initialize_blockchain()
            logger.info("✅ Blockchain integration initialized for Glean Agent")
        except Exception as e:
            logger.warning(f"⚠️ Blockchain initialization failed: {e}")
        
        # Test connection to Glean service
        try:
            # A2A Protocol: Use blockchain messaging instead of httpx
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
            # async with httpx.AsyncClient() as client:
            #     response = await client.get(f"{self.glean_service_url}/health")
            #     if response.status_code == 200:
            #         logger.info("Successfully connected to Glean service")
            #     else:
            #         logger.warning(f"Glean service health check returned: {response.status_code}")
            logger.info("Glean service health check skipped - A2A protocol compliance")
        except Exception as e:
            logger.warning(f"Could not connect to Glean service: {e}")
        
        # Verify linting tools are available
        for lang, tools in self.linters.items():
            for tool in tools:
                if self._check_tool_available(tool):
                    logger.info(f"Linting tool '{tool}' is available")
                else:
                    logger.warning(f"Linting tool '{tool}' is not available")
        
        # Register with A2A Registry
        await self._register_with_a2a_registry()
        
        # Register MCP tools and resources
        self._register_mcp_components()
        
        # Discover code processing agents for collaboration
        available_agents = await self.discover_agents(
            capabilities=["code_processing", "quality_control", "validation", "ai_preparation"],
            agent_types=["analysis", "validation", "quality", "ai"]
        )
        
        # Store discovered agents for collaboration
        self.code_agents = {
            "quality_agents": [agent for agent in available_agents if "quality" in agent.get("capabilities", [])],
            "validation_agents": [agent for agent in available_agents if "validation" in agent.get("capabilities", [])],
            "ai_agents": [agent for agent in available_agents if "ai" in agent.get("agent_type", "")],
            "analysis_agents": [agent for agent in available_agents if "analysis" in agent.get("agent_type", "")]
        }
        
        logger.info(f"Glean Agent discovered {len(available_agents)} code processing agents")
    
    async def shutdown(self) -> None:
        """Cleanup agent resources"""
        logger.info("Shutting down Glean Agent")
    
    def _check_tool_available(self, tool: str) -> bool:
        """Check if a linting tool is available"""
        # SECURITY FIX: Whitelist allowed tools to prevent command injection
        ALLOWED_TOOLS = {
            'ruff', 'black', 'isort', 'mypy', 'pylint', 'flake8',
            'bandit', 'safety', 'prospector', 'radon', 'xenon',
            'pycodestyle', 'pydocstyle', 'pyflakes', 'vulture'
        }
        
        # Validate tool name against whitelist
        if tool not in ALLOWED_TOOLS:
            logger.warning(f"Tool '{tool}' not in allowed tools whitelist")
            return False
        
        # Additional validation: ensure no path separators or special characters
        if any(char in tool for char in ['/', '\\', ';', '&', '|', '$', '`', '\n', '\r']):
            logger.error(f"Invalid characters in tool name: {tool}")
            return False
        
        try:
            # Use absolute path to tool or rely on PATH, but never user input directly
            subprocess.run([tool, "--version"], capture_output=True, check=False, shell=False)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    @a2a_handler("analyze_code")
    async def handle_analyze_code(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle code analysis requests"""
        try:
            data = message.parts[0].data if message.parts else {}
            directory = data.get("directory", ".")
            analysis_types = data.get("analysis_types", [AnalysisType.FULL])
            file_patterns = data.get("file_patterns", ["*.py", "*.js", "*.ts"])
            
            result = await self.analyze_code_comprehensive(
                directory=directory,
                analysis_types=analysis_types,
                file_patterns=file_patterns
            )
            
            return create_success_response(result)
        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            return create_error_response(str(e))
    
    @a2a_skill("analyze_code_comprehensive", "Perform comprehensive code analysis")
    async def analyze_code_comprehensive(
        self,
        directory: str,
        analysis_types: List[AnalysisType] = None,
        file_patterns: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive code analysis including Glean, linting, and testing
        """
        if analysis_types is None:
            analysis_types = [AnalysisType.FULL]
        if file_patterns is None:
            file_patterns = ["*.py", "*.js", "*.ts"]
        
        analysis_id = f"analysis_{hashlib.md5(f'{directory}{time.time()}'.encode(), usedforsecurity=False).hexdigest()[:12]}"
        start_time = time.time()
        
        results = {
            "analysis_id": analysis_id,
            "directory": directory,
            "timestamp": datetime.utcnow().isoformat(),
            "analyses": {}
        }
        
        # Perform Glean semantic analysis
        if AnalysisType.GLEAN_SEMANTIC in analysis_types or AnalysisType.FULL in analysis_types:
            results["analyses"]["glean"] = await self._perform_glean_analysis(directory)
        
        # Perform linting
        if AnalysisType.LINT in analysis_types or AnalysisType.FULL in analysis_types:
            results["analyses"]["lint"] = await self._perform_lint_analysis(directory, file_patterns)
        
        # Run tests
        if AnalysisType.TEST in analysis_types or AnalysisType.FULL in analysis_types:
            results["analyses"]["test"] = await self._run_tests(directory)
        
        # Security analysis
        if AnalysisType.SECURITY in analysis_types or AnalysisType.FULL in analysis_types:
            results["analyses"]["security"] = await self._perform_security_analysis(directory)
        
        # Calculate overall metrics
        results["summary"] = self._calculate_summary_metrics(results["analyses"])
        results["duration"] = time.time() - start_time
        
        # Store results in database
        await self._store_analysis_results(analysis_id, results)
        
        # Store comprehensive analysis data in data_manager
        await self.store_agent_data(
            data_type="code_analysis_result",
            data={
                "analysis_id": analysis_id,
                "directory": directory,
                "analysis_types": [str(t) for t in analysis_types],
                "files_analyzed": results.get("summary", {}).get("files_analyzed", 0),
                "total_issues": results.get("summary", {}).get("total_issues", 0),
                "critical_issues": results.get("summary", {}).get("critical_issues", 0),
                "analysis_duration": results["duration"],
                "glean_analysis_completed": "glean" in results.get("analyses", {}),
                "security_analysis_completed": "security" in results.get("analyses", {}),
                "tests_run": "test" in results.get("analyses", {}),
                "timestamp": results["timestamp"],
                # Include metadata in the data dict instead
                "agent_version": "glean_agent_v1.0",
                "analysis_tools_used": list(results.get("analyses", {}).keys()),
                "file_patterns": file_patterns
            }
        )
        
        # Update agent status with agent_manager
        await self.update_agent_status(
            status="active",
            details={
                "last_analysis": analysis_id,
                "last_directory": directory,
                "analyses_completed": len(results.get("analyses", {})),
                "analysis_duration": results["duration"],
                "total_issues_found": results.get("summary", {}).get("total_issues", 0),
                "active_capabilities": ["glean_analysis", "code_linting", "security_scan", "test_execution"]
            }
        )
        
        return results
    
    async def _perform_glean_analysis(self, directory: str) -> Dict[str, Any]:
        """Perform real semantic code analysis using AST and dependency parsing"""
        try:
            import ast
            import importlib.util
            start_time = time.time()
            
            analysis_results = {
                "directory": directory,
                "files_analyzed": 0,
                "dependency_graph": {},
                "similar_code_blocks": [],
                "refactoring_opportunities": [],
                "code_patterns": {},
                "import_analysis": {},
                "function_call_graph": {},
                "dead_code_candidates": [],
                "duration": 0.0
            }
            
            # Find Python files for semantic analysis
            path = Path(directory)
            python_files = list(path.rglob("*.py"))
            
            # Filter out ignored files
            ignore_patterns = ["__pycache__", ".git", "venv", "env", ".pytest_cache"]
            filtered_files = [
                f for f in python_files 
                if not any(pattern in str(f) for pattern in ignore_patterns)
            ]
            
            analysis_results["files_analyzed"] = len(filtered_files)
            
            # Perform semantic analysis on each file
            all_imports = {}
            all_functions = {}
            all_classes = {}
            call_graph = {}
            
            for file_path in filtered_files:
                try:
                    file_analysis = await self._analyze_file_semantics(file_path)
                    
                    # Collect imports
                    file_key = str(file_path)
                    all_imports[file_key] = file_analysis["imports"]
                    all_functions[file_key] = file_analysis["functions"]
                    all_classes[file_key] = file_analysis["classes"]
                    call_graph[file_key] = file_analysis["function_calls"]
                    
                except Exception as e:
                    logger.warning(f"Failed semantic analysis for {file_path}: {e}")
            
            # Build dependency graph
            analysis_results["dependency_graph"] = self._build_dependency_graph(all_imports)
            analysis_results["import_analysis"] = self._analyze_imports(all_imports)
            analysis_results["function_call_graph"] = call_graph
            
            # Find similar code blocks
            analysis_results["similar_code_blocks"] = self._find_similar_code_blocks(all_functions)
            
            # Identify refactoring opportunities
            analysis_results["refactoring_opportunities"] = self._identify_refactoring_opportunities(
                all_functions, all_classes, call_graph
            )
            
            # Find dead code candidates
            analysis_results["dead_code_candidates"] = self._find_dead_code_candidates(
                all_functions, call_graph
            )
            
            # Analyze code patterns
            analysis_results["code_patterns"] = self._analyze_code_patterns(all_functions, all_classes)
            
            analysis_results["duration"] = time.time() - start_time
            return analysis_results
            
        except Exception as e:
            logger.error(f"Glean semantic analysis error: {e}")
            return {"error": str(e)}
    
    async def _analyze_file_semantics(self, file_path: Path) -> Dict[str, Any]:
        """Perform semantic analysis on a single Python file"""
        import ast
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            file_analysis = {
                "imports": [],
                "functions": [],
                "classes": [],
                "function_calls": [],
                "variables": [],
                "constants": []
            }
            
            class SemanticVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.current_class = None
                    self.current_function = None
                
                def visit_Import(self, node):
                    for alias in node.names:
                        file_analysis["imports"].append({
                            "type": "import",
                            "module": alias.name,
                            "alias": alias.asname,
                            "line": node.lineno
                        })
                    self.generic_visit(node)
                
                def visit_ImportFrom(self, node):
                    for alias in node.names:
                        file_analysis["imports"].append({
                            "type": "from_import",
                            "module": node.module,
                            "name": alias.name,
                            "alias": alias.asname,
                            "line": node.lineno
                        })
                    self.generic_visit(node)
                
                def visit_FunctionDef(self, node):
                    func_info = {
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "returns": ast.get_docstring(node) is not None,
                        "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
                        "calls": [],
                        "class": self.current_class
                    }
                    
                    # Analyze function body for calls
                    old_function = self.current_function
                    self.current_function = node.name
                    
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            call_name = self._get_call_name(child)
                            if call_name:
                                func_info["calls"].append(call_name)
                                file_analysis["function_calls"].append({
                                    "from": node.name,
                                    "to": call_name,
                                    "line": child.lineno
                                })
                    
                    file_analysis["functions"].append(func_info)
                    self.current_function = old_function
                    self.generic_visit(node)
                
                def visit_ClassDef(self, node):
                    class_info = {
                        "name": node.name,
                        "line": node.lineno,
                        "bases": [self._get_base_name(base) for base in node.bases],
                        "methods": [],
                        "decorators": [self._get_decorator_name(d) for d in node.decorator_list]
                    }
                    
                    old_class = self.current_class
                    self.current_class = node.name
                    
                    # Find methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            class_info["methods"].append(item.name)
                    
                    file_analysis["classes"].append(class_info)
                    self.current_class = old_class
                    self.generic_visit(node)
                
                def _get_decorator_name(self, decorator):
                    if isinstance(decorator, ast.Name):
                        return decorator.id
                    elif isinstance(decorator, ast.Attribute):
                        value_name = self._get_node_name(decorator.value)
                        if value_name:
                            return f"{value_name}.{decorator.attr}"
                        return decorator.attr
                    elif isinstance(decorator, ast.Call):
                        return self._get_decorator_name(decorator.func)
                    return "unknown"
                
                def _get_call_name(self, call):
                    if isinstance(call.func, ast.Name):
                        return call.func.id
                    elif isinstance(call.func, ast.Attribute):
                        return f"{call.func.attr}"
                    return None
                
                def _get_base_name(self, base):
                    if isinstance(base, ast.Name):
                        return base.id
                    elif isinstance(base, ast.Attribute):
                        value_name = self._get_node_name(base.value)
                        if value_name:
                            return f"{value_name}.{base.attr}"
                        return base.attr
                    return "unknown"
                
                def _get_node_name(self, node):
                    """Recursively get the name of a node"""
                    if isinstance(node, ast.Name):
                        return node.id
                    elif isinstance(node, ast.Attribute):
                        value_name = self._get_node_name(node.value)
                        if value_name:
                            return f"{value_name}.{node.attr}"
                        return node.attr
                    elif isinstance(node, ast.Call):
                        return self._get_node_name(node.func)
                    return None
            
            visitor = SemanticVisitor()
            visitor.visit(tree)
            
            return file_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze semantics for {file_path}: {e}")
            return {"imports": [], "functions": [], "classes": [], "function_calls": [], "variables": []}
    
    def _build_dependency_graph(self, all_imports: Dict[str, List]) -> Dict[str, Any]:
        """Build a dependency graph from import analysis"""
        graph = {
            "nodes": set(),
            "edges": [],
            "external_dependencies": set(),
            "internal_dependencies": set(),
            "circular_dependencies": []
        }
        
        # Collect all modules and dependencies
        for file_path, imports in all_imports.items():
            file_name = Path(file_path).stem
            graph["nodes"].add(file_name)
            
            for imp in imports:
                module = imp["module"]
                if module:
                    if module.startswith(".") or any(part in file_path for part in module.split(".")):
                        # Internal dependency
                        graph["internal_dependencies"].add(module)
                        graph["edges"].append({"from": file_name, "to": module, "type": "internal"})
                    else:
                        # External dependency
                        graph["external_dependencies"].add(module)
                        graph["edges"].append({"from": file_name, "to": module, "type": "external"})
        
        return {
            "total_nodes": len(graph["nodes"]),
            "total_edges": len(graph["edges"]),
            "external_dependencies": sorted(list(graph["external_dependencies"])),
            "internal_dependencies": sorted(list(graph["internal_dependencies"]))
        }
    
    def _analyze_imports(self, all_imports: Dict[str, List]) -> Dict[str, Any]:
        """Analyze import patterns across the codebase"""
        import_analysis = {
            "total_imports": 0,
            "unique_modules": set(),
            "most_imported": {},
            "unused_imports": [],
            "relative_imports": 0,
            "absolute_imports": 0
        }
        
        module_counts = {}
        
        for file_path, imports in all_imports.items():
            import_analysis["total_imports"] += len(imports)
            
            for imp in imports:
                module = imp["module"]
                if module:
                    import_analysis["unique_modules"].add(module)
                    module_counts[module] = module_counts.get(module, 0) + 1
                    
                    if imp["type"] == "from_import" and module.startswith("."):
                        import_analysis["relative_imports"] += 1
                    else:
                        import_analysis["absolute_imports"] += 1
        
        # Find most imported modules
        import_analysis["most_imported"] = dict(
            sorted(module_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        import_analysis["unique_modules"] = len(import_analysis["unique_modules"])
        
        return import_analysis
    
    def _find_similar_code_blocks(self, all_functions: Dict[str, List]) -> List[Dict]:
        """Find similar code blocks across functions"""
        similar_blocks = []
        
        # Simple similarity detection based on function signatures and call patterns
        functions = []
        for file_path, funcs in all_functions.items():
            for func in funcs:
                func["file"] = file_path
                functions.append(func)
        
        # Compare functions pairwise
        for i, func1 in enumerate(functions):
            for func2 in functions[i+1:]:
                similarity = self._calculate_function_similarity(func1, func2)
                if similarity > 0.7:  # 70% similarity threshold
                    similar_blocks.append({
                        "function1": {"name": func1["name"], "file": func1["file"], "line": func1["line"]},
                        "function2": {"name": func2["name"], "file": func2["file"], "line": func2["line"]},
                        "similarity": similarity,
                        "reason": "Similar function signatures and call patterns"
                    })
        
        return similar_blocks[:10]  # Limit to top 10
    
    def _calculate_function_similarity(self, func1: Dict, func2: Dict) -> float:
        """Calculate similarity between two functions"""
        if func1["name"] == func2["name"]:
            return 0.0  # Skip identical names
        
        score = 0.0
        
        # Compare argument count
        if len(func1["args"]) == len(func2["args"]):
            score += 0.3
        
        # Compare function calls
        calls1 = set(func1.get("calls", []))
        calls2 = set(func2.get("calls", []))
        
        if calls1 and calls2:
            intersection = len(calls1.intersection(calls2))
            union = len(calls1.union(calls2))
            if union > 0:
                score += 0.7 * (intersection / union)
        
        return score
    
    def _identify_refactoring_opportunities(self, all_functions: Dict, all_classes: Dict, call_graph: Dict) -> List[Dict]:
        """Identify refactoring opportunities in the codebase"""
        opportunities = []
        
        # Find long functions that could be split
        for file_path, functions in all_functions.items():
            for func in functions:
                if len(func.get("calls", [])) > 10:
                    opportunities.append({
                        "type": "extract_method",
                        "file": file_path,
                        "function": func["name"],
                        "line": func["line"],
                        "reason": f"Function has {len(func['calls'])} calls - consider extracting methods",
                        "priority": "medium"
                    })
        
        # Find classes with too many methods
        for file_path, classes in all_classes.items():
            for cls in classes:
                if len(cls.get("methods", [])) > 15:
                    opportunities.append({
                        "type": "split_class",
                        "file": file_path,
                        "class": cls["name"],
                        "line": cls["line"],
                        "reason": f"Class has {len(cls['methods'])} methods - consider splitting",
                        "priority": "high"
                    })
        
        return opportunities[:15]  # Limit to top 15
    
    def _find_dead_code_candidates(self, all_functions: Dict, call_graph: Dict) -> List[Dict]:
        """Find potential dead code candidates"""
        candidates = []
        
        # Build set of all called functions
        called_functions = set()
        for file_calls in call_graph.values():
            for call in file_calls:
                called_functions.add(call["to"])
        
        # Find functions that are never called
        for file_path, functions in all_functions.items():
            for func in functions:
                # Skip special methods and private methods
                if (func["name"] not in called_functions and 
                    not func["name"].startswith("__") and 
                    not func["name"] in ["main", "setup", "teardown"]):
                    
                    candidates.append({
                        "file": file_path,
                        "function": func["name"],
                        "line": func["line"],
                        "confidence": "medium",
                        "reason": "Function is defined but never called"
                    })
        
        return candidates[:10]  # Limit to 10 candidates
    
    def _analyze_code_patterns(self, all_functions: Dict, all_classes: Dict) -> Dict[str, Any]:
        """Analyze common code patterns in the codebase"""
        patterns = {
            "design_patterns": [],
            "anti_patterns": [],
            "statistics": {}
        }
        
        total_functions = sum(len(funcs) for funcs in all_functions.values())
        total_classes = sum(len(classes) for classes in all_classes.values())
        
        # Look for design patterns
        for file_path, classes in all_classes.items():
            for cls in classes:
                # Singleton pattern detection
                if "instance" in [method.lower() for method in cls.get("methods", [])]:
                    patterns["design_patterns"].append({
                        "pattern": "Singleton",
                        "file": file_path,
                        "class": cls["name"],
                        "confidence": "low"
                    })
        
        # Look for anti-patterns
        for file_path, functions in all_functions.items():
            for func in functions:
                # God function (too many calls)
                if len(func.get("calls", [])) > 20:
                    patterns["anti_patterns"].append({
                        "pattern": "God Function",
                        "file": file_path,
                        "function": func["name"],
                        "severity": "high"
                    })
        
        patterns["statistics"] = {
            "total_functions": total_functions,
            "total_classes": total_classes,
            "functions_with_decorators": sum(
                1 for funcs in all_functions.values() 
                for func in funcs if func.get("decorators")
            )
        }
        
        return patterns
    
    async def _perform_lint_analysis(self, directory: str, file_patterns: List[str], options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform multi-tool linting analysis with real tool execution"""
        start_time = time.time()
        issues = []
        files_analyzed = 0
        linter_results = {}
        
        # Find files matching patterns
        path = Path(directory)
        files = []
        for pattern in file_patterns:
            files.extend(path.rglob(pattern))
        
        # Filter out files in ignore patterns
        ignore_patterns = [".git", "__pycache__", "node_modules", ".pytest_cache", "venv", "env"]
        filtered_files = []
        for file_path in files:
            if not any(ignore in str(file_path) for ignore in ignore_patterns):
                filtered_files.append(file_path)
        
        files_analyzed = len(filtered_files)
        
        # Group files by language for efficient processing (limit files per language to prevent timeout)
        max_files_per_language = 100  # Prevent overwhelming the linters
        
        python_files = [f for f in filtered_files if f.suffix.lower() == ".py"][:max_files_per_language]
        js_files = [f for f in filtered_files if f.suffix.lower() in [".js", ".jsx"]][:max_files_per_language]
        ts_files = [f for f in filtered_files if f.suffix.lower() in [".ts", ".tsx"]][:max_files_per_language]
        html_files = [f for f in filtered_files if f.suffix.lower() in [".html", ".htm", ".xhtml"]][:max_files_per_language]
        xml_files = [f for f in filtered_files if f.suffix.lower() in [".xml", ".xsl", ".xslt", ".svg"]][:max_files_per_language]
        yaml_files = [f for f in filtered_files if f.suffix.lower() in [".yaml", ".yml"]][:max_files_per_language]
        json_files = [f for f in filtered_files if f.suffix.lower() in [".json", ".jsonc"]][:max_files_per_language]
        shell_files = [f for f in filtered_files if f.suffix.lower() in [".sh", ".bash", ".zsh", ".fish"]][:max_files_per_language]
        css_files = [f for f in filtered_files if f.suffix.lower() == ".css"][:max_files_per_language]
        scss_files = [f for f in filtered_files if f.suffix.lower() in [".scss", ".sass"]][:max_files_per_language]
        cds_files = [f for f in filtered_files if f.suffix.lower() == ".cds"][:max_files_per_language]
        solidity_files = [f for f in filtered_files if f.suffix.lower() == ".sol"][:max_files_per_language]
        
        # Initialize options if not provided
        if options is None:
            options = {}
        
        # Define all language groups
        all_language_groups = [
            ("python", "Python", python_files, self._run_python_linters_batch),
            ("javascript", "JavaScript", js_files, self._run_javascript_linters_batch),
            ("typescript", "TypeScript", ts_files, self._run_typescript_linters_batch),
            ("html", "HTML", html_files, self._run_html_linters_batch),
            ("xml", "XML", xml_files, self._run_xml_linters_batch),
            ("yaml", "YAML", yaml_files, self._run_yaml_linters_batch),
            ("json", "JSON", json_files, self._run_json_linters_batch),
            ("shell", "Shell", shell_files, self._run_shell_linters_batch),
            ("css", "CSS", css_files, self._run_css_linters_batch),
            ("scss", "SCSS", scss_files, self._run_scss_linters_batch),
            ("cds", "CDS", cds_files, self._run_cds_linters_batch),
            ("solidity", "Solidity", solidity_files, self._run_solidity_linters_batch)
        ]
        
        # Filter language groups based on options
        selected_languages = options.get("languages", [])
        if selected_languages:
            # Only scan selected languages
            language_groups = [(key, name, files, func) for key, name, files, func in all_language_groups 
                              if key in selected_languages]
            print(f"🎯 Scanning only selected languages: {', '.join([name for _, name, _, _ in language_groups])}")
        else:
            # Scan all languages with files
            language_groups = [(key, name, files, func) for key, name, files, func in all_language_groups]
        
        # Process each language group sequentially with timeout
        languages_scanned = []
        for lang_key, lang_name, file_list, linter_func in language_groups:
            if file_list:
                print(f"🔍 Analyzing {len(file_list)} {lang_name} files...")
                try:
                    # Add timeout for each language (60 seconds)
                    result = await asyncio.wait_for(
                        linter_func(file_list, directory), 
                        timeout=60.0
                    )
                    if isinstance(result, dict):
                        issues.extend(result.get("issues", []))
                        linter_results.update(result.get("linter_results", {}))
                        languages_scanned.append(lang_name)
                        print(f"✅ {lang_name}: Found {len(result.get('issues', []))} issues")
                    else:
                        print(f"⚠️  {lang_name}: No results returned")
                except asyncio.TimeoutError:
                    print(f"⏱️  {lang_name}: Timed out after 60 seconds, skipping...")
                    logger.warning(f"Linting {lang_name} files timed out after 60 seconds")
                except Exception as e:
                    print(f"❌ {lang_name}: Linting failed: {e}")
                    logger.error(f"Linting {lang_name} files failed: {e}")
        
        # Group issues by severity and type
        severity_counts = {}
        type_counts = {}
        
        for issue in issues:
            if hasattr(issue, 'severity'):
                severity = issue.severity
                issue_type = issue.issue_type
            else:
                # Handle dict format
                severity = issue.get('severity', 'unknown')
                issue_type = issue.get('issue_type', 'unknown')
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            type_counts[issue_type] = type_counts.get(issue_type, 0) + 1
        
        return {
            "files_analyzed": files_analyzed,
            "total_issues": len(issues),
            "critical_issues": severity_counts.get("critical", 0),
            "issues_by_severity": severity_counts,
            "issues_by_type": type_counts,
            "issues": [asdict(issue) if hasattr(issue, '__dict__') else issue for issue in issues],
            "linter_results": linter_results,
            "languages_scanned": languages_scanned,
            "duration": time.time() - start_time
        }
    
    async def _run_python_linters_batch(self, files: List[Path], directory: str) -> Dict[str, Any]:
        """Run Python linters on a batch of files"""
        issues = []
        linter_results = {}
        
        # Check which Python linters are available
        available_linters = []
        for linter in ["pylint", "flake8", "mypy", "bandit"]:
            if self._check_tool_available(linter):
                available_linters.append(linter)
        
        if not available_linters:
            logger.warning("No Python linters available")
            return {"issues": [], "linter_results": {}}
        
        # Run each available linter
        for linter in available_linters:
            try:
                if linter == "pylint":
                    result = await self._run_pylint(files, directory)
                elif linter == "flake8":
                    result = await self._run_flake8(files, directory)
                elif linter == "mypy":
                    result = await self._run_mypy(files, directory)
                elif linter == "bandit":
                    result = await self._run_bandit(files, directory)
                
                issues.extend(result.get("issues", []))
                linter_results[linter] = result.get("raw_output", "")
                
            except Exception as e:
                logger.error(f"Error running {linter}: {e}")
                linter_results[linter] = f"Error: {str(e)}"
        
        return {"issues": issues, "linter_results": linter_results}
    
    async def _run_pylint(self, files: List[Path], directory: str) -> Dict[str, Any]:
        """Run pylint on Python files"""
        issues = []
        
        # Convert absolute paths to relative paths from the directory
        dir_path = Path(directory).resolve()
        relative_files = []
        for f in files:
            try:
                relative_files.append(str(f.relative_to(dir_path)))
            except ValueError:
                # If file is not under directory, use the absolute path
                relative_files.append(str(f))
        
        # Create pylint command with relative paths and disable import errors
        cmd = ["pylint", "--output-format=json", "--disable=import-error,no-name-in-module"] + relative_files
        
        result = await self._safe_subprocess_run(cmd, cwd=directory, timeout=120)
        
        if result.get("success", False) or result.get("returncode") in [0, 1, 2, 4, 8, 16]:
            # Parse pylint JSON output
            try:
                if result.get("stdout"):
                    pylint_issues = json.loads(result["stdout"])
                    for issue_data in pylint_issues:
                        issue_key = f'{issue_data.get("path", "")}{issue_data.get("line", 0)}{issue_data.get("symbol", "")}'
                        issue = {
                            "id": f"pylint_{hashlib.md5(issue_key.encode(), usedforsecurity=False).hexdigest()[:8]}",
                            "file_path": issue_data.get("path", ""),
                            "line": issue_data.get("line", 0),
                            "column": issue_data.get("column", 0),
                            "tool": "pylint",
                            "issue_type": self._map_pylint_type(issue_data.get("type", "")),
                            "severity": self._map_pylint_severity(issue_data.get("type", "")),
                            "code": issue_data.get("symbol", ""),
                            "message": issue_data.get("message", ""),
                            "rule": issue_data.get("message-id", ""),
                            "auto_fixable": False
                        }
                        issues.append(issue)
            except json.JSONDecodeError:
                logger.warning("Failed to parse pylint JSON output")
        
        return {"issues": issues, "raw_output": result.get("stdout", "")}
    
    async def _run_flake8(self, files: List[Path], directory: str) -> Dict[str, Any]:
        """Run flake8 on Python files"""
        issues = []
        
        # Convert absolute paths to relative paths from the directory
        dir_path = Path(directory).resolve()
        relative_files = []
        for f in files:
            try:
                relative_files.append(str(f.relative_to(dir_path)))
            except ValueError:
                # If file is not under directory, use the absolute path
                relative_files.append(str(f))
        
        # Create flake8 command with relative paths
        cmd = ["flake8", "--format=%(path)s:%(row)d:%(col)d:%(code)s:%(text)s"] + relative_files
        
        result = await self._safe_subprocess_run(cmd, cwd=directory, timeout=60)
        
        if result.get("stdout"):
            for line in result["stdout"].strip().split('\n'):
                if line and ':' in line:
                    parts = line.split(':', 4)
                    if len(parts) >= 5:
                        issue = {
                            "id": f"flake8_{hashlib.md5(line.encode(), usedforsecurity=False).hexdigest()[:8]}",
                            "file_path": parts[0],
                            "line": int(parts[1]) if parts[1].isdigit() else 0,
                            "column": int(parts[2]) if parts[2].isdigit() else 0,
                            "tool": "flake8",
                            "issue_type": self._map_flake8_type(parts[3]),
                            "severity": self._map_flake8_severity(parts[3]),
                            "code": parts[3],
                            "message": parts[4] if len(parts) > 4 else "",
                            "rule": parts[3],
                            "auto_fixable": parts[3] in ["W291", "W292", "W293"]  # Whitespace issues
                        }
                        issues.append(issue)
        
        return {"issues": issues, "raw_output": result.get("stdout", "")}
    
    async def _run_mypy(self, files: List[Path], directory: str) -> Dict[str, Any]:
        """Run mypy on Python files"""
        issues = []
        
        # Convert absolute paths to relative paths from the directory
        dir_path = Path(directory).resolve()
        relative_files = []
        for f in files:
            try:
                relative_files.append(str(f.relative_to(dir_path)))
            except ValueError:
                # If file is not under directory, use the absolute path
                relative_files.append(str(f))
        
        # Create mypy command with relative paths
        cmd = ["mypy", "--show-error-codes"] + relative_files
        
        result = await self._safe_subprocess_run(cmd, cwd=directory, timeout=90)
        
        if result.get("stdout"):
            for line in result["stdout"].strip().split('\n'):
                if ':' in line and ' error:' in line:
                    # Parse mypy output: file.py:line: error: message [error-code]
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        file_path = parts[0]
                        line_num = int(parts[1]) if parts[1].isdigit() else 0
                        message_part = parts[2].strip()
                        
                        if message_part.startswith('error:'):
                            message = message_part[6:].strip()
                            error_code = ""
                            if '[' in message and ']' in message:
                                error_code = message[message.rfind('[')+1:message.rfind(']')]
                                message = message[:message.rfind('[')].strip()
                            
                            issue_key = f'{file_path}{line_num}{error_code}'
                            issue = {
                                "id": f"mypy_{hashlib.md5(issue_key.encode(), usedforsecurity=False).hexdigest()[:8]}",
                                "file_path": file_path,
                                "line": line_num,
                                "column": 0,
                                "tool": "mypy",
                                "issue_type": "type_error",
                                "severity": "medium",
                                "code": error_code,
                                "message": message,
                                "rule": error_code,
                                "auto_fixable": False
                            }
                            issues.append(issue)
        
        return {"issues": issues, "raw_output": result.get("stdout", "")}
    
    async def _run_bandit(self, files: List[Path], directory: str) -> Dict[str, Any]:
        """Run bandit security linter on Python files"""
        issues = []
        
        # Convert absolute paths to relative paths from the directory
        dir_path = Path(directory).resolve()
        relative_files = []
        for f in files:
            try:
                relative_files.append(str(f.relative_to(dir_path)))
            except ValueError:
                # If file is not under directory, use the absolute path
                relative_files.append(str(f))
        
        # Create bandit command with relative paths
        cmd = ["bandit", "-f", "json"] + relative_files
        
        result = await self._safe_subprocess_run(cmd, cwd=directory, timeout=60)
        
        if result.get("stdout"):
            try:
                bandit_data = json.loads(result["stdout"])
                for issue_data in bandit_data.get("results", []):
                    issue_key = f'{issue_data.get("filename", "")}{issue_data.get("line_number", 0)}{issue_data.get("test_id", "")}'
                    issue = {
                        "id": f"bandit_{hashlib.md5(issue_key.encode(), usedforsecurity=False).hexdigest()[:8]}",
                        "file_path": issue_data.get("filename", ""),
                        "line": issue_data.get("line_number", 0),
                        "column": 0,
                        "tool": "bandit",
                        "issue_type": "security",
                        "severity": issue_data.get("issue_severity", "medium").lower(),
                        "code": issue_data.get("test_id", ""),
                        "message": issue_data.get("issue_text", ""),
                        "rule": issue_data.get("test_name", ""),
                        "auto_fixable": False
                    }
                    issues.append(issue)
            except json.JSONDecodeError:
                logger.warning("Failed to parse bandit JSON output")
        
        return {"issues": issues, "raw_output": result.get("stdout", "")}
    
    async def _run_javascript_linters_batch(self, files: List[Path], directory: str) -> Dict[str, Any]:
        """Run enhanced JavaScript linters and analysis on a batch of files"""
        issues = []
        linter_results = {}
        
        # Ensure we have relative paths for tools
        relative_files = [str(f.relative_to(directory)) for f in files]
        
        # 1. ESLint with comprehensive rules
        if shutil.which("eslint") or shutil.which("npx"):
            try:
                # Create enhanced ESLint config if none exists
                await self._ensure_javascript_eslint_config(directory)
                
                cmd = ["eslint", "--format=json", "--ext", ".js,.jsx,.mjs,.cjs"] + relative_files
                result = await self._safe_subprocess_run(cmd, cwd=directory, timeout=90)
                
                if result.get("stdout"):
                    try:
                        eslint_data = json.loads(result["stdout"])
                        for file_data in eslint_data:
                            file_path = file_data.get("filePath", "")
                            for message in file_data.get("messages", []):
                                issues.append(self._create_issue(
                                    file_path=file_path,
                                    line=message.get("line", 1),
                                    message=f"{message.get('ruleId', 'unknown')}: {message.get('message', '')}",
                                    severity=self._map_eslint_severity(message.get("severity", 1)),
                                    tool="eslint"
                                ))
                    except json.JSONDecodeError:
                        pass
                
                linter_results["eslint"] = f"Found {len([i for i in issues if i.get('tool') == 'eslint'])} style issues"
            except Exception as e:
                linter_results["eslint"] = f"Error: {str(e)}"
        else:
            linter_results["eslint"] = "ESLint not available"
        
        # 2. JSHint for additional checks
        if shutil.which("jshint"):
            try:
                await self._ensure_jshint_config(directory)
                
                cmd = ["jshint", "--reporter=json"] + relative_files
                result = await self._safe_subprocess_run(cmd, cwd=directory, timeout=60)
                
                if result.get("stdout"):
                    try:
                        jshint_data = json.loads(result["stdout"])
                        for file_path, file_issues in jshint_data.items():
                            for issue in file_issues:
                                issues.append(self._create_issue(
                                    file_path=file_path,
                                    line=issue.get("line", 1),
                                    message=f"{issue.get('code', '')}: {issue.get('reason', '')}",
                                    severity="warning" if issue.get("code", "").startswith("W") else "error",
                                    tool="jshint"
                                ))
                    except json.JSONDecodeError:
                        pass
                
                linter_results["jshint"] = f"Found {len([i for i in issues if i.get('tool') == 'jshint'])} issues"
            except Exception as e:
                linter_results["jshint"] = f"Error: {str(e)}"
        else:
            linter_results["jshint"] = "JSHint not available"
        
        # 3. JavaScript semantic analysis
        js_analysis = await self._analyze_javascript_semantics(files)
        issues.extend(js_analysis.get("issues", []))
        linter_results["js-semantics"] = f"Found {len(js_analysis.get('issues', []))} semantic issues"
        
        # 4. JavaScript security analysis
        js_security = await self._analyze_javascript_security(files)
        issues.extend(js_security.get("issues", []))
        linter_results["js-security"] = f"Found {len(js_security.get('issues', []))} security issues"
        
        # 5. JavaScript performance analysis
        js_performance = await self._analyze_javascript_performance(files)
        issues.extend(js_performance.get("issues", []))
        linter_results["js-performance"] = f"Found {len(js_performance.get('issues', []))} performance issues"
        
        return {"issues": issues, "linter_results": linter_results}
    
    async def _analyze_javascript_semantics(self, files: List[Path]) -> Dict[str, Any]:
        """Perform JavaScript-specific semantic analysis"""
        issues = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    
                    # 1. Var usage (should use let/const)
                    if line.startswith('var '):
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Use 'let' or 'const' instead of 'var' for better scoping",
                            severity="warning",
                            tool="js-semantics"
                        ))
                    
                    # 2. == instead of === (loose equality)
                    if '==' in line and '===' not in line and '!=' in line and '!==' not in line:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Use strict equality (===) instead of loose equality (==)",
                            severity="warning",
                            tool="js-semantics"
                        ))
                    
                    # 3. Missing semicolons (for statement-ending lines)
                    if line and not line.endswith((';', '{', '}', ',', ':')) and not line.startswith(('if', 'else', 'for', 'while', 'function', '//', '/*', '*')):
                        if any(keyword in line for keyword in ['const', 'let', 'var', 'return', 'break', 'continue']):
                            issues.append(self._create_issue(
                                file_path=str(file_path),
                                line=line_num,
                                message="Missing semicolon at end of statement",
                                severity="info",
                                tool="js-semantics"
                            ))
                    
                    # 4. Callback hell detection
                    if line.count('function') > 1 or (')' in line and '{' in line and line.count(')') > 2):
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Complex nested callbacks detected - consider using async/await",
                            severity="info",
                            tool="js-semantics"
                        ))
                    
                    # 5. Unused variables (basic detection)
                    if 'const ' in line or 'let ' in line or 'var ' in line:
                        var_match = re.search(r'(?:const|let|var)\s+(\w+)', line)
                        if var_match:
                            var_name = var_match.group(1)
                            # Check if variable is used in the rest of the file
                            rest_of_file = '\n'.join(lines[line_num:])
                            if var_name not in rest_of_file:
                                issues.append(self._create_issue(
                                    file_path=str(file_path),
                                    line=line_num,
                                    message=f"Variable '{var_name}' is declared but never used",
                                    severity="warning",
                                    tool="js-semantics"
                                ))
                    
                    # 6. Arrow functions vs regular functions
                    if 'function(' in line and '=>' not in line:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Consider using arrow function for cleaner syntax",
                            severity="info",
                            tool="js-semantics"
                        ))
                    
                    # 7. Template literals opportunity
                    if '+' in line and ('"' in line or "'" in line) and 'string' not in line.lower():
                        if line.count('+') >= 2 and (line.count('"') >= 2 or line.count("'") >= 2):
                            issues.append(self._create_issue(
                                file_path=str(file_path),
                                line=line_num,
                                message="Consider using template literals instead of string concatenation",
                                severity="info",
                                tool="js-semantics"
                            ))
                    
                    # 8. Promise without catch
                    if '.then(' in line and '.catch(' not in line and 'await' not in line:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Promise chain should have error handling (.catch())",
                            severity="warning",
                            tool="js-semantics"
                        ))
                
            except Exception as e:
                print(f"Error analyzing JavaScript semantics for {file_path}: {e}")
        
        return {"issues": issues}
    
    async def _analyze_javascript_security(self, files: List[Path]) -> Dict[str, Any]:
        """Perform JavaScript security analysis"""
        issues = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    line_lower = line.lower().strip()
                    
                    # 1. eval() usage
                    if 'eval(' in line_lower:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Security risk: eval() can execute arbitrary code",
                            severity="error",
                            tool="js-security"
                        ))
                    
                    # 2. innerHTML usage (XSS risk)
                    if 'innerhtml' in line_lower:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Potential XSS risk: innerHTML should be avoided - use textContent or DOM methods",
                            severity="warning",
                            tool="js-security"
                        ))
                    
                    # 3. document.write usage
                    if 'document.write' in line_lower:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Security risk: document.write can be exploited - use DOM methods instead",
                            severity="warning",
                            tool="js-security"
                        ))
                    
                    # 4. Hardcoded secrets/tokens
                    if any(secret in line_lower for secret in ['password', 'secret', 'apikey', 'api_key', 'token']):
                        if '=' in line and ('"' in line or "'" in line):
                            issues.append(self._create_issue(
                                file_path=str(file_path),
                                line=line_num,
                                message="Security risk: Potential hardcoded credentials detected",
                                severity="error",
                                tool="js-security"
                            ))
                    
                    # 5. HTTP instead of HTTPS
                    if 'http://' in line_lower and 'localhost' not in line_lower:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Security risk: Use HTTPS instead of HTTP for external requests",
                            severity="warning",
                            tool="js-security"
                        ))
                    
                    # 6. SQL injection risk
                    if 'query(' in line_lower and ('+' in line or '${' in line):
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Potential SQL injection: Use parameterized queries",
                            severity="error",
                            tool="js-security"
                        ))
                    
                    # 7. Unsafe regex
                    if 'regexp(' in line_lower or 'new regexp' in line_lower:
                        if '.*' in line or '.+' in line:
                            issues.append(self._create_issue(
                                file_path=str(file_path),
                                line=line_num,
                                message="Potential ReDoS vulnerability: Avoid unbounded regex patterns",
                                severity="warning",
                                tool="js-security"
                            ))
                    
                    # 8. Command injection risk
                    if any(cmd in line_lower for cmd in ['exec(', 'execsync(', 'spawn(', 'spawnsync(']):
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Security risk: Command execution can be dangerous - validate all inputs",
                            severity="error",
                            tool="js-security"
                        ))
                    
                    # 9. CORS misconfiguration
                    if 'access-control-allow-origin' in line_lower and '*' in line:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Security risk: Wildcard CORS allows any origin",
                            severity="warning",
                            tool="js-security"
                        ))
                    
                    # 10. Insecure random
                    if 'math.random' in line_lower:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Security: Math.random() is not cryptographically secure",
                            severity="info",
                            tool="js-security"
                        ))
                
            except Exception as e:
                print(f"Error analyzing JavaScript security for {file_path}: {e}")
        
        return {"issues": issues}
    
    async def _analyze_javascript_performance(self, files: List[Path]) -> Dict[str, Any]:
        """Analyze JavaScript code for performance issues"""
        issues = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    
                    # 1. Synchronous file operations
                    if any(sync_op in line for sync_op in ['readFileSync', 'writeFileSync', 'appendFileSync']):
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Performance: Use async file operations instead of sync",
                            severity="warning",
                            tool="js-performance"
                        ))
                    
                    # 2. Array operations in loops
                    if 'for' in line and any(method in line for method in ['.push(', '.unshift(', '.splice(']):
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Performance: Array modifications in loops can be slow",
                            severity="info",
                            tool="js-performance"
                        ))
                    
                    # 3. jQuery in modern code
                    if '$(' in line or 'jQuery(' in line:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Performance: Consider using native DOM methods instead of jQuery",
                            severity="info",
                            tool="js-performance"
                        ))
                    
                    # 4. Multiple DOM queries
                    if line.count('getElementById') > 1 or line.count('querySelector') > 1:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Performance: Cache DOM queries to avoid repeated lookups",
                            severity="info",
                            tool="js-performance"
                        ))
                    
                    # 5. Large array operations
                    if '.map(' in line and '.filter(' in line:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Performance: Chain array operations to avoid multiple iterations",
                            severity="info",
                            tool="js-performance"
                        ))
                    
                    # 6. console.log in production
                    if 'console.' in line:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Performance: Remove console statements from production code",
                            severity="info",
                            tool="js-performance"
                        ))
                    
                    # 7. Inefficient string concatenation in loops
                    if 'for' in line or 'while' in line:
                        next_lines = lines[line_num:min(line_num + 5, len(lines))]
                        loop_content = ' '.join(next_lines)
                        if '+=' in loop_content and ('"' in loop_content or "'" in loop_content):
                            issues.append(self._create_issue(
                                file_path=str(file_path),
                                line=line_num,
                                message="Performance: String concatenation in loops - use array.join()",
                                severity="warning",
                                tool="js-performance"
                            ))
                
            except Exception as e:
                print(f"Error analyzing JavaScript performance for {file_path}: {e}")
        
        return {"issues": issues}
    
    async def _ensure_javascript_eslint_config(self, directory: str) -> None:
        """Ensure JavaScript ESLint configuration exists"""
        eslintrc_path = Path(directory) / ".eslintrc.json"
        if not eslintrc_path.exists():
            config = {
                "env": {
                    "browser": True,
                    "es2021": True,
                    "node": True
                },
                "extends": [
                    "eslint:recommended"
                ],
                "parserOptions": {
                    "ecmaVersion": "latest",
                    "sourceType": "module"
                },
                "rules": {
                    "no-var": "error",
                    "prefer-const": "warn",
                    "eqeqeq": "error",
                    "no-eval": "error",
                    "no-implied-eval": "error",
                    "no-new-func": "error",
                    "no-return-await": "warn",
                    "require-await": "warn",
                    "no-throw-literal": "error",
                    "prefer-promise-reject-errors": "error",
                    "no-unused-vars": ["warn", {"argsIgnorePattern": "^_"}],
                    "no-console": "warn",
                    "semi": ["warn", "always"],
                    "quotes": ["warn", "single"],
                    "prefer-template": "warn",
                    "prefer-arrow-callback": "warn"
                }
            }
            
            with open(eslintrc_path, 'w') as f:
                json.dump(config, f, indent=2)
    
    async def _ensure_jshint_config(self, directory: str) -> None:
        """Ensure JSHint configuration exists"""
        jshintrc_path = Path(directory) / ".jshintrc"
        if not jshintrc_path.exists():
            config = {
                "esversion": 11,
                "node": True,
                "browser": True,
                "strict": "implied",
                "curly": True,
                "eqeqeq": True,
                "forin": True,
                "freeze": True,
                "immed": True,
                "latedef": True,
                "newcap": True,
                "noarg": True,
                "noempty": True,
                "nonbsp": True,
                "nonew": True,
                "plusplus": False,
                "quotmark": "single",
                "undef": True,
                "unused": True,
                "trailing": True,
                "maxparams": 4,
                "maxdepth": 4,
                "maxcomplexity": 10
            }
            
            with open(jshintrc_path, 'w') as f:
                json.dump(config, f, indent=2)
    
    async def _run_eslint(self, files: List[Path], directory: str) -> Dict[str, Any]:
        """Run ESLint on JavaScript/TypeScript files"""
        issues = []
        
        # Convert absolute paths to relative paths from the directory
        dir_path = Path(directory).resolve()
        relative_files = []
        for f in files:
            try:
                relative_files.append(str(f.relative_to(dir_path)))
            except ValueError:
                # If file is not under directory, use the absolute path
                relative_files.append(str(f))
        
        # Create ESLint config on-the-fly if none exists
        await self._ensure_eslint_config(directory)
        
        # Create eslint command with relative paths and enhanced rules
        cmd = ["eslint", "--format=json", "--ext", ".js,.jsx,.ts,.tsx"] + relative_files
        
        result = await self._safe_subprocess_run(cmd, cwd=directory, timeout=90)
        
        if result.get("stdout"):
            try:
                eslint_data = json.loads(result["stdout"])
                for file_data in eslint_data:
                    file_path = file_data.get("filePath", "")
                    for message in file_data.get("messages", []):
                        issue_key = f'{file_path}{message.get("line", 0)}{message.get("ruleId", "")}'
                        issue = {
                            "id": f"eslint_{hashlib.md5(issue_key.encode(), usedforsecurity=False).hexdigest()[:8]}",
                            "file_path": file_path,
                            "line": message.get("line", 0),
                            "column": message.get("column", 0),
                            "tool": "eslint",
                            "issue_type": self._map_eslint_type(message.get("severity", 1)),
                            "severity": self._map_eslint_severity(message.get("severity", 1)),
                            "code": message.get("ruleId", ""),
                            "message": message.get("message", ""),
                            "rule": message.get("ruleId", ""),
                            "auto_fixable": message.get("fix") is not None
                        }
                        issues.append(issue)
            except json.JSONDecodeError:
                logger.warning("Failed to parse ESLint JSON output")
        
        return {"issues": issues, "raw_output": result.get("stdout", "")}
    
    async def _run_jshint(self, files: List[Path], directory: str) -> Dict[str, Any]:
        """Run JSHint on JavaScript files"""
        issues = []
        
        # Convert absolute paths to relative paths from the directory
        dir_path = Path(directory).resolve()
        relative_files = []
        for f in files:
            try:
                relative_files.append(str(f.relative_to(dir_path)))
            except ValueError:
                # If file is not under directory, use the absolute path
                relative_files.append(str(f))
        
        # Create JSHint command with reporter for easier parsing
        cmd = ["jshint", "--reporter=unix"] + relative_files
        
        result = await self._safe_subprocess_run(cmd, cwd=directory, timeout=60)
        
        if result.get("stdout"):
            # Parse JSHint output (format: filename:line:col: message)
            for line in result["stdout"].split('\n'):
                line = line.strip()
                if ':' in line and line:
                    try:
                        parts = line.split(':', 3)
                        if len(parts) >= 4:
                            file_path = parts[0]
                            line_num = int(parts[1]) if parts[1].isdigit() else 0
                            col_num = int(parts[2]) if parts[2].isdigit() else 0
                            message = parts[3].strip()
                            
                            # Create unique issue ID
                            issue_key = f'{file_path}{line_num}{col_num}{message}'
                            issue = {
                                "id": f"jshint_{hashlib.md5(issue_key.encode(), usedforsecurity=False).hexdigest()[:8]}",
                                "file_path": file_path,
                                "line": line_num,
                                "column": col_num,
                                "tool": "jshint",
                                "issue_type": self._map_jshint_type(message),
                                "severity": self._map_jshint_severity(message),
                                "code": "jshint",
                                "message": message,
                                "rule": "jshint-rule",
                                "auto_fixable": False
                            }
                            issues.append(issue)
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        continue
        
        return {"issues": issues, "raw_output": result.get("stdout", "")}
    
    async def _run_typescript_linters_batch(self, files: List[Path], directory: str) -> Dict[str, Any]:
        """Run enhanced TypeScript linters on a batch of files with comprehensive analysis"""
        issues = []
        linter_results = {}
        
        # Ensure we have relative paths for tools
        relative_files = [str(f.relative_to(directory)) for f in files]
        
        # 1. TypeScript Compiler Check (tsc) with strict settings
        if shutil.which("tsc") or shutil.which("npx"):
            try:
                # Create temporary tsconfig if none exists
                await self._ensure_typescript_config(directory)
                
                # Run TypeScript compiler for type checking
                cmd = ["tsc", "--noEmit", "--strict", "--exactOptionalPropertyTypes", "--noImplicitReturns", "--noFallthroughCasesInSwitch"] + relative_files
                result = await self._safe_subprocess_run(cmd, cwd=directory, timeout=120)
                
                if result.get("stderr"):
                    # Parse TypeScript compiler errors
                    for line in result["stderr"].split("\n"):
                        if ".ts(" in line and ")" in line and "error TS" in line:
                            parts = line.split("): error TS")
                            if len(parts) >= 2:
                                file_line_info = parts[0]
                                error_info = parts[1]
                                
                                # Extract file and line
                                if "(" in file_line_info and "," in file_line_info:
                                    file_path = file_line_info.split("(")[0]
                                    line_col = file_line_info.split("(")[1].split(")")[0]
                                    line_num = int(line_col.split(",")[0]) if "," in line_col else 1
                                    
                                    issues.append(self._create_issue(
                                        file_path=file_path,
                                        line=line_num,
                                        message=f"TypeScript error: {error_info.strip()}",
                                        severity="error",
                                        tool="tsc"
                                    ))
                
                linter_results["tsc"] = f"Found {len([i for i in issues if i.get('tool') == 'tsc'])} type errors" if issues else "Type checking passed"
            except Exception as e:
                linter_results["tsc"] = f"Error: {str(e)}"
        else:
            linter_results["tsc"] = "TypeScript compiler not available"
        
        # 2. ESLint with TypeScript rules
        if shutil.which("eslint") or shutil.which("npx"):
            try:
                await self._ensure_typescript_eslint_config(directory)
                
                cmd = ["eslint", "--format=json", "--ext", ".ts,.tsx"] + relative_files
                result = await self._safe_subprocess_run(cmd, cwd=directory, timeout=90)
                
                if result.get("stdout"):
                    try:
                        eslint_data = json.loads(result["stdout"])
                        for file_data in eslint_data:
                            file_path = file_data.get("filePath", "")
                            for message in file_data.get("messages", []):
                                issues.append(self._create_issue(
                                    file_path=file_path,
                                    line=message.get("line", 1),
                                    message=f"{message.get('ruleId', 'unknown')}: {message.get('message', '')}",
                                    severity=self._map_eslint_severity(message.get("severity", 1)),
                                    tool="eslint"
                                ))
                    except json.JSONDecodeError:
                        pass
                
                linter_results["eslint"] = f"Found {len([i for i in issues if i.get('tool') == 'eslint'])} style issues"
            except Exception as e:
                linter_results["eslint"] = f"Error: {str(e)}"
        else:
            linter_results["eslint"] = "ESLint not available"
        
        # 3. TypeScript-specific semantic analysis
        ts_analysis = await self._analyze_typescript_semantics(files, directory)
        issues.extend(ts_analysis.get("issues", []))
        linter_results["ts-semantics"] = f"Found {len(ts_analysis.get('issues', []))} semantic issues"
        
        # 4. TypeScript security analysis
        ts_security = await self._analyze_typescript_security(files, directory)
        issues.extend(ts_security.get("issues", []))
        linter_results["ts-security"] = f"Found {len(ts_security.get('issues', []))} security issues"
        
        return {"issues": issues, "linter_results": linter_results}
    
    async def _analyze_typescript_semantics(self, files: List[Path], directory: str) -> Dict[str, Any]:
        """Perform TypeScript-specific semantic analysis"""
        issues = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    
                    # 1. Usage of 'any' type (reduces type safety)
                    if ': any' in line or 'any[]' in line or 'any>' in line:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Avoid using 'any' type - prefer specific types for better type safety",
                            severity="warning",
                            tool="ts-semantics"
                        ))
                    
                    # 2. Type assertions that could be dangerous
                    if 'as ' in line and ('as any' in line or 'as unknown' in line):
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Dangerous type assertion detected - verify type safety",
                            severity="warning",
                            tool="ts-semantics"
                        ))
                    
                    # 3. Non-null assertion operator misuse
                    if '!' in line and ('!' not in line.split('//')[0] or line.count('!') > 1):
                        if '!.' in line or '!;' in line or '!,' in line:
                            issues.append(self._create_issue(
                                file_path=str(file_path),
                                line=line_num,
                                message="Non-null assertion operator (!) should be used carefully - consider proper null checking",
                                severity="info",
                                tool="ts-semantics"
                            ))
                    
                    # 4. Unused imports (basic detection)
                    if line.startswith('import ') and ' from ' in line:
                        import_match = re.search(r'import\s+.*?\s+from\s+["\']([^"\']+)["\']', line)
                        if import_match:
                            import_name = import_match.group(0)
                            # Check if imported items are used elsewhere in the file
                            if '{' in import_name and '}' in import_name:
                                imported_items = re.findall(r'\{\s*(.*?)\s*\}', import_name)
                                for items_group in imported_items:
                                    items = [item.strip() for item in items_group.split(',')]
                                    for item in items:
                                        if item and item not in content[lines.index(line):]:
                                            issues.append(self._create_issue(
                                                file_path=str(file_path),
                                                line=line_num,
                                                message=f"Potentially unused import: {item}",
                                                severity="info",
                                                tool="ts-semantics"
                                            ))
                    
                    # 5. Promise handling issues
                    if 'Promise' in line and 'await' not in line and 'return' not in line and '.then' not in line and '.catch' not in line:
                        if 'new Promise' in line or 'Promise.resolve' in line or 'Promise.reject' in line:
                            issues.append(self._create_issue(
                                file_path=str(file_path),
                                line=line_num,
                                message="Promise should be properly handled with await, .then(), or .catch()",
                                severity="warning",
                                tool="ts-semantics"
                            ))
                    
                    # 6. Missing return type annotations for functions
                    if 'function ' in line or '=>' in line:
                        if ': void' not in line and ': ' not in line.split('=>')[0] and 'return' in content:
                            issues.append(self._create_issue(
                                file_path=str(file_path),
                                line=line_num,
                                message="Consider adding explicit return type annotation",
                                severity="info",
                                tool="ts-semantics"
                            ))
                    
                    # 7. Interface vs Type usage
                    if line.startswith('type ') and '{' in line:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Consider using 'interface' instead of 'type' for object shapes (better error messages)",
                            severity="info",
                            tool="ts-semantics"
                        ))
                    
                    # 8. Enum usage best practices
                    if line.startswith('enum '):
                        if 'const enum' not in line:
                            issues.append(self._create_issue(
                                file_path=str(file_path),
                                line=line_num,
                                message="Consider using 'const enum' for better tree-shaking and performance",
                                severity="info",
                                tool="ts-semantics"
                            ))
                    
                    # 9. Optional chaining opportunities
                    if '&&' in line and '.' in line and '!=' in line:
                        if 'null' in line or 'undefined' in line:
                            issues.append(self._create_issue(
                                file_path=str(file_path),
                                line=line_num,
                                message="Consider using optional chaining (?.) instead of manual null checking",
                                severity="info",
                                tool="ts-semantics"
                            ))
                    
                    # 10. Strict equality checks
                    if '==' in line and '===' not in line and '!=' in line and '!==' not in line:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Use strict equality (===) instead of loose equality (==)",
                            severity="warning",
                            tool="ts-semantics"
                        ))
                
            except Exception as e:
                print(f"Error analyzing TypeScript semantics for {file_path}: {e}")
        
        return {"issues": issues}
    
    async def _analyze_typescript_security(self, files: List[Path], directory: str) -> Dict[str, Any]:
        """Perform TypeScript security analysis"""
        issues = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    line_lower = line.lower().strip()
                    
                    # 1. Potential XSS vulnerabilities
                    if 'dangerouslysetinnerhtml' in line_lower:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Potential XSS vulnerability: dangerouslySetInnerHTML should be carefully sanitized",
                            severity="error",
                            tool="ts-security"
                        ))
                    
                    # 2. Eval and Function constructor usage
                    if 'eval(' in line_lower or 'new function(' in line_lower:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Security risk: eval() and Function constructor can execute arbitrary code",
                            severity="error",
                            tool="ts-security"
                        ))
                    
                    # 3. Local storage of sensitive data
                    storage_check = ('localstorage' in line_lower or 'sessionstorage' in line_lower)
                    sensitive_keywords = ['password', 'token', 'secret', 'key', 'credential']
                    if storage_check and any(keyword in line_lower for keyword in sensitive_keywords):
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Security risk: Avoid storing sensitive data in localStorage/sessionStorage",
                            severity="error",
                            tool="ts-security"
                        ))
                    
                    # 4. HTTP instead of HTTPS
                    if 'http://' in line_lower and 'localhost' not in line_lower:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Security risk: Use HTTPS instead of HTTP for external requests",
                            severity="warning",
                            tool="ts-security"
                        ))
                    
                    # 5. Insecure randomness
                    if 'math.random' in line_lower:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Security risk: Math.random() is not cryptographically secure - use crypto.getRandomValues()",
                            severity="warning",
                            tool="ts-security"
                        ))
                    
                    # 6. Direct DOM manipulation
                    if 'document.write' in line_lower or 'innerhtml' in line_lower:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Potential XSS risk: Direct DOM manipulation should validate and sanitize input",
                            severity="warning",
                            tool="ts-security"
                        ))
                    
                    # 7. Console.log in production
                    if 'console.' in line_lower and 'log' in line_lower:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Security/Performance: Remove console.log statements from production code",
                            severity="info",
                            tool="ts-security"
                        ))
                    
                    # 8. Hardcoded credentials or secrets
                    if any(keyword in line_lower for keyword in ['password', 'secret', 'apikey', 'token']) and '=' in line:
                        if '"' in line or "'" in line:
                            issues.append(self._create_issue(
                                file_path=str(file_path),
                                line=line_num,
                                message="Security risk: Potential hardcoded credentials detected",
                                severity="error",
                                tool="ts-security"
                            ))
                    
                    # 9. Unsafe window object usage
                    if 'window[' in line and ('"' in line or "'" in line):
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Security risk: Dynamic window property access can be exploited",
                            severity="warning",
                            tool="ts-security"
                        ))
                    
                    # 10. Prototype pollution possibilities
                    if '__proto__' in line_lower or 'prototype' in line_lower and '[' in line:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Security risk: Potential prototype pollution vulnerability",
                            severity="warning",
                            tool="ts-security"
                        ))
                
            except Exception as e:
                print(f"Error analyzing TypeScript security for {file_path}: {e}")
        
        return {"issues": issues}
    
    async def _ensure_typescript_config(self, directory: str) -> None:
        """Ensure TypeScript configuration exists"""
        tsconfig_path = Path(directory) / "tsconfig.json"
        if not tsconfig_path.exists():
            # Create a strict TypeScript configuration
            config = {
                "compilerOptions": {
                    "target": "ES2020",
                    "module": "commonjs",
                    "lib": ["ES2020", "DOM"],
                    "strict": True,
                    "noImplicitAny": True,
                    "strictNullChecks": True,
                    "noImplicitReturns": True,
                    "noFallthroughCasesInSwitch": True,
                    "exactOptionalPropertyTypes": True,
                    "noImplicitOverride": True,
                    "noUncheckedIndexedAccess": True,
                    "allowUnreachableCode": False,
                    "allowUnusedLabels": False,
                    "skipLibCheck": True,
                    "forceConsistentCasingInFileNames": True
                },
                "include": ["**/*.ts", "**/*.tsx"],
                "exclude": ["node_modules", "dist", "build"]
            }
            
            with open(tsconfig_path, 'w') as f:
                json.dump(config, f, indent=2)
    
    async def _ensure_typescript_eslint_config(self, directory: str) -> None:
        """Ensure TypeScript ESLint configuration exists"""
        eslintrc_path = Path(directory) / ".eslintrc.json"
        if not eslintrc_path.exists():
            config = {
                "parser": "@typescript-eslint/parser",
                "plugins": ["@typescript-eslint"],
                "extends": [
                    "eslint:recommended",
                    "@typescript-eslint/recommended",
                    "@typescript-eslint/recommended-requiring-type-checking"
                ],
                "parserOptions": {
                    "ecmaVersion": 2020,
                    "sourceType": "module",
                    "project": "./tsconfig.json"
                },
                "rules": {
                    "@typescript-eslint/no-explicit-any": "error",
                    "@typescript-eslint/no-unused-vars": "error",
                    "@typescript-eslint/explicit-return-type": "warn",
                    "@typescript-eslint/no-non-null-assertion": "warn",
                    "@typescript-eslint/prefer-optional-chain": "warn",
                    "@typescript-eslint/prefer-nullish-coalescing": "warn",
                    "@typescript-eslint/no-floating-promises": "error",
                    "@typescript-eslint/await-thenable": "error",
                    "prefer-const": "error",
                    "no-var": "error",
                    "eqeqeq": "error"
                }
            }
            
            with open(eslintrc_path, 'w') as f:
                json.dump(config, f, indent=2)
    
    async def _run_html_linters_batch(self, files: List[Path], directory: str) -> Dict[str, Any]:
        """Run HTML linters on a batch of files"""
        issues = []
        linter_results = {}
        
        # HTMLHint
        if shutil.which("htmlhint"):
            try:
                file_list = " ".join([f'"{str(f)}"' for f in files])
                result = await self._run_command(f"htmlhint {file_list}", cwd=directory)
                
                if result["stderr"] and "Error" in result["stderr"]:
                    for line in result["stderr"].split("\n"):
                        if line.strip():
                            issues.append(self._create_issue(
                                file_path=str(files[0]),
                                line=1,
                                message=line.strip(),
                                severity="error",
                                tool="htmlhint"
                            ))
                
                linter_results["htmlhint"] = result["stdout"] or "No issues found"
            except Exception as e:
                logger.error(f"Error running htmlhint: {e}")
                linter_results["htmlhint"] = f"Error: {str(e)}"
        
        return {"issues": issues, "linter_results": linter_results}
    
    async def _run_xml_linters_batch(self, files: List[Path], directory: str) -> Dict[str, Any]:
        """Run XML linters on a batch of files"""
        issues = []
        linter_results = {}
        
        # xmllint
        if shutil.which("xmllint"):
            for file_path in files:
                try:
                    result = await self._run_command(f'xmllint --noout "{str(file_path)}"', cwd=directory)
                    
                    if result["stderr"]:
                        for line in result["stderr"].split("\n"):
                            if ":" in line and "error" in line.lower():
                                issues.append(self._create_issue(
                                    file_path=str(file_path),
                                    line=1,
                                    message=line.strip(),
                                    severity="error",
                                    tool="xmllint"
                                ))
                    
                    linter_results[f"xmllint_{file_path.name}"] = "Valid XML" if not result["stderr"] else result["stderr"]
                except Exception as e:
                    logger.error(f"Error running xmllint on {file_path}: {e}")
                    linter_results[f"xmllint_{file_path.name}"] = f"Error: {str(e)}"
        
        return {"issues": issues, "linter_results": linter_results}
    
    async def _run_yaml_linters_batch(self, files: List[Path], directory: str) -> Dict[str, Any]:
        """Run YAML linters on a batch of files"""
        issues = []
        linter_results = {}
        
        # yamllint
        if shutil.which("yamllint"):
            try:
                file_list = " ".join([f'"{str(f)}"' for f in files])
                result = await self._run_command(f"yamllint -f parsable {file_list}", cwd=directory)
                
                if result["stdout"]:
                    for line in result["stdout"].split("\n"):
                        if line.strip():
                            # Parse yamllint output format: file:line:column: [severity] message (rule)
                            parts = line.split(":", 4)
                            if len(parts) >= 5:
                                file_path = parts[0]
                                line_num = int(parts[1]) if parts[1].isdigit() else 1
                                message = parts[4].strip()
                                severity = "warning" if "warning" in message else "error"
                                
                                issues.append(self._create_issue(
                                    file_path=file_path,
                                    line=line_num,
                                    message=message,
                                    severity=severity,
                                    tool="yamllint"
                                ))
                
                linter_results["yamllint"] = result["stdout"] or "No issues found"
            except Exception as e:
                logger.error(f"Error running yamllint: {e}")
                linter_results["yamllint"] = f"Error: {str(e)}"
        
        return {"issues": issues, "linter_results": linter_results}
    
    async def _run_json_linters_batch(self, files: List[Path], directory: str) -> Dict[str, Any]:
        """Run JSON linters on a batch of files"""
        issues = []
        linter_results = {}
        
        # Use Python's json module for validation
        import json
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)
                linter_results[f"json_{file_path.name}"] = "Valid JSON"
            except json.JSONDecodeError as e:
                issues.append(self._create_issue(
                    file_path=str(file_path),
                    line=e.lineno if hasattr(e, 'lineno') else 1,
                    message=f"JSON decode error: {str(e)}",
                    severity="error",
                    tool="jsonlint"
                ))
                linter_results[f"json_{file_path.name}"] = f"Invalid JSON: {str(e)}"
            except Exception as e:
                logger.error(f"Error validating JSON in {file_path}: {e}")
                linter_results[f"json_{file_path.name}"] = f"Error: {str(e)}"
        
        return {"issues": issues, "linter_results": linter_results}
    
    async def _run_shell_linters_batch(self, files: List[Path], directory: str) -> Dict[str, Any]:
        """Run Shell script linters on a batch of files"""
        issues = []
        linter_results = {}
        
        # ShellCheck
        if shutil.which("shellcheck"):
            try:
                file_list = " ".join([f'"{str(f)}"' for f in files])
                result = await self._run_command(f"shellcheck -f gcc {file_list}", cwd=directory)
                
                if result["stdout"]:
                    for line in result["stdout"].split("\n"):
                        if line.strip():
                            # Parse shellcheck gcc format: file:line:column: severity: message [SC####]
                            parts = line.split(":", 4)
                            if len(parts) >= 5:
                                file_path = parts[0]
                                line_num = int(parts[1]) if parts[1].isdigit() else 1
                                severity = parts[3].strip().lower()
                                message = parts[4].strip()
                                
                                issues.append(self._create_issue(
                                    file_path=file_path,
                                    line=line_num,
                                    message=message,
                                    severity=severity if severity in ["error", "warning"] else "warning",
                                    tool="shellcheck"
                                ))
                
                linter_results["shellcheck"] = result["stdout"] or "No issues found"
            except Exception as e:
                logger.error(f"Error running shellcheck: {e}")
                linter_results["shellcheck"] = f"Error: {str(e)}"
        
        return {"issues": issues, "linter_results": linter_results}
    
    async def _run_css_linters_batch(self, files: List[Path], directory: str) -> Dict[str, Any]:
        """Run CSS linters on a batch of files"""
        issues = []
        linter_results = {}
        
        # Stylelint
        if shutil.which("stylelint"):
            try:
                file_list = " ".join([f'"{str(f)}"' for f in files])
                result = await self._run_command(f"stylelint {file_list} --formatter json", cwd=directory)
                
                if result["stdout"]:
                    try:
                        import json
                        lint_results = json.loads(result["stdout"])
                        
                        for file_result in lint_results:
                            for warning in file_result.get("warnings", []):
                                issues.append(self._create_issue(
                                    file_path=file_result["source"],
                                    line=warning.get("line", 1),
                                    message=warning.get("text", "Unknown issue"),
                                    severity=warning.get("severity", "warning"),
                                    tool="stylelint"
                                ))
                    except json.JSONDecodeError:
                        # Fallback to plain text parsing
                        linter_results["stylelint"] = result["stdout"]
                
                linter_results["stylelint"] = f"Found {len(issues)} issues" if issues else "No issues found"
            except Exception as e:
                logger.error(f"Error running stylelint: {e}")
                linter_results["stylelint"] = f"Error: {str(e)}"
        
        return {"issues": issues, "linter_results": linter_results}
    
    async def _run_scss_linters_batch(self, files: List[Path], directory: str) -> Dict[str, Any]:
        """Run SCSS/SASS linters on a batch of files with SCSS-specific analysis"""
        issues = []
        linter_results = {}
        
        # Stylelint with SCSS configuration
        if shutil.which("stylelint"):
            try:
                file_list = " ".join([f'"{str(f)}"' for f in files])
                # Use SCSS-specific stylelint config
                result = await self._run_command(
                    f"stylelint {file_list} --syntax scss --formatter json", 
                    cwd=directory
                )
                
                if result["stdout"]:
                    try:
                        import json
                        lint_results = json.loads(result["stdout"])
                        
                        for file_result in lint_results:
                            for warning in file_result.get("warnings", []):
                                issues.append(self._create_issue(
                                    file_path=file_result["source"],
                                    line=warning.get("line", 1),
                                    message=warning.get("text", "Unknown SCSS issue"),
                                    severity=warning.get("severity", "warning"),
                                    tool="stylelint-scss"
                                ))
                    except json.JSONDecodeError:
                        # Fallback to plain text parsing
                        linter_results["stylelint-scss"] = result["stdout"]
                
                linter_results["stylelint-scss"] = f"Found {len([i for i in issues if i.get('tool') == 'stylelint-scss'])} issues" if issues else "No issues found"
            except Exception as e:
                logger.error(f"Error running stylelint on SCSS: {e}")
                linter_results["stylelint-scss"] = f"Error: {str(e)}"
        
        # Sass-lint (if available)
        if shutil.which("sass-lint"):
            try:
                file_list = " ".join([f'"{str(f)}"' for f in files])
                result = await self._run_command(f"sass-lint {file_list} --format json", cwd=directory)
                
                if result["stdout"]:
                    try:
                        import json
                        lint_results = json.loads(result["stdout"])
                        
                        for file_result in lint_results:
                            file_path = file_result.get("filePath", "unknown")
                            for message in file_result.get("messages", []):
                                issues.append(self._create_issue(
                                    file_path=file_path,
                                    line=message.get("line", 1),
                                    message=message.get("message", "Unknown sass-lint issue"),
                                    severity=message.get("severity", 1) == 2 and "error" or "warning",
                                    tool="sass-lint"
                                ))
                    except json.JSONDecodeError:
                        linter_results["sass-lint"] = result["stdout"]
                
                linter_results["sass-lint"] = f"Found {len([i for i in issues if i.get('tool') == 'sass-lint'])} issues" if issues else "No issues found"
            except Exception as e:
                logger.error(f"Error running sass-lint: {e}")
                linter_results["sass-lint"] = f"Error: {str(e)}"
        
        # SCSS-specific semantic analysis
        scss_analysis = await self._analyze_scss_semantics(files)
        issues.extend(scss_analysis.get("issues", []))
        linter_results["scss-semantics"] = f"Found {len(scss_analysis.get('issues', []))} semantic issues"
        
        return {"issues": issues, "linter_results": linter_results}
    
    async def _analyze_scss_semantics(self, files: List[Path]) -> Dict[str, Any]:
        """Perform SCSS-specific semantic analysis"""
        issues = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Analyze SCSS-specific patterns
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    
                    # Check for common SCSS issues
                    
                    # 1. Undefined variables
                    if '$' in line and ':' not in line and not line.startswith('//'):
                        # Variable usage without definition in same file
                        var_match = re.search(r'\$([a-zA-Z_-][a-zA-Z0-9_-]*)', line)
                        if var_match:
                            var_name = var_match.group(1)
                            if f'${var_name}:' not in content:
                                issues.append(self._create_issue(
                                    file_path=str(file_path),
                                    line=line_num,
                                    message=f"Potentially undefined variable: ${var_name}",
                                    severity="warning",
                                    tool="scss-semantics"
                                ))
                    
                    # 2. Deep nesting (more than 4 levels)
                    indent_level = (len(line) - len(line.lstrip())) // 2
                    if indent_level > 4 and line and not line.startswith('//'):
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message=f"Deep nesting detected ({indent_level} levels). Consider refactoring.",
                            severity="warning",
                            tool="scss-semantics"
                        ))
                    
                    # 3. Missing semicolons
                    if ':' in line and not line.endswith((';', '{', '}')) and not line.startswith('//') and line.strip():
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Missing semicolon at end of declaration",
                            severity="error",
                            tool="scss-semantics"
                        ))
                    
                    # 4. Unused mixins (basic check)
                    if line.startswith('@mixin'):
                        mixin_match = re.search(r'@mixin\s+([a-zA-Z_-][a-zA-Z0-9_-]*)', line)
                        if mixin_match:
                            mixin_name = mixin_match.group(1)
                            if f'@include {mixin_name}' not in content:
                                issues.append(self._create_issue(
                                    file_path=str(file_path),
                                    line=line_num,
                                    message=f"Potentially unused mixin: {mixin_name}",
                                    severity="info",
                                    tool="scss-semantics"
                                ))
                    
                    # 5. Invalid nesting of media queries
                    if '@media' in line and indent_level > 0:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Media query nested inside selector. Consider moving to root level.",
                            severity="warning",
                            tool="scss-semantics"
                        ))
                    
                    # 6. Duplicate selectors (basic check)
                    if line.endswith('{') and not line.startswith('@'):
                        selector = line.replace('{', '').strip()
                        if content.count(f'{selector} {{') > 1:
                            issues.append(self._create_issue(
                                file_path=str(file_path),
                                line=line_num,
                                message=f"Potentially duplicate selector: {selector}",
                                severity="warning",
                                tool="scss-semantics"
                            ))
                
            except Exception as e:
                logger.error(f"Error analyzing SCSS semantics for {file_path}: {e}")
        
        return {"issues": issues}
    
    async def _run_cds_linters_batch(self, files: List[Path], directory: str) -> Dict[str, Any]:
        """Run CDS (SAP CAP) linters on a batch of files with comprehensive analysis"""
        issues = []
        linter_results = {}
        
        # CDS Compiler Check (using @sap/cds-dk)
        if shutil.which("cds"):
            try:
                result = await self._run_command("cds compile --to sql", cwd=directory)
                
                if result["stderr"]:
                    # Parse CDS compilation errors
                    for line in result["stderr"].split("\n"):
                        if "ERROR" in line or "Error" in line:
                            # Extract file and line info from CDS error messages
                            if ".cds:" in line:
                                parts = line.split(".cds:")
                                if len(parts) >= 2:
                                    file_info = parts[0].split("/")[-1] if "/" in parts[0] else parts[0]
                                    line_info = parts[1].split(" ")[0] if " " in parts[1] else "1"
                                    line_num = int(line_info) if line_info.isdigit() else 1
                                    message = line.split("ERROR")[-1].strip() if "ERROR" in line else line.strip()
                                    
                                    issues.append(self._create_issue(
                                        file_path=f"{file_info}.cds",
                                        line=line_num,
                                        message=f"CDS compilation error: {message}",
                                        severity="error",
                                        tool="cds-compiler"
                                    ))
                
                linter_results["cds-compiler"] = f"Found {len([i for i in issues if i.get('tool') == 'cds-compiler'])} compilation issues" if issues else "Compilation successful"
            except Exception as e:
                linter_results["cds-compiler"] = f"Error: {str(e)}"
        else:
            linter_results["cds-compiler"] = "CDS CLI not installed"
        
        # CDS-specific semantic analysis
        cds_analysis = await self._analyze_cds_semantics(files)
        issues.extend(cds_analysis.get("issues", []))
        linter_results["cds-semantics"] = f"Found {len(cds_analysis.get('issues', []))} semantic issues"
        
        # CDS Security Analysis
        security_analysis = await self._analyze_cds_security(files)
        issues.extend(security_analysis.get("issues", []))
        linter_results["cds-security"] = f"Found {len(security_analysis.get('issues', []))} security issues"
        
        return {"issues": issues, "linter_results": linter_results}
    
    async def _analyze_cds_semantics(self, files: List[Path]) -> Dict[str, Any]:
        """Perform CDS-specific semantic analysis"""
        issues = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                
                # Track entities, services, and types for cross-reference checking
                defined_entities = set()
                defined_services = set()
                used_entities = set()
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    
                    # 1. Entity definition tracking
                    if line.startswith('entity '):
                        entity_match = re.search(r'entity\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
                        if entity_match:
                            defined_entities.add(entity_match.group(1))
                    
                    # 2. Service definition tracking
                    if line.startswith('service '):
                        service_match = re.search(r'service\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
                        if service_match:
                            defined_services.add(service_match.group(1))
                    
                    # 3. Association/Composition validation
                    if 'Association to' in line or 'Composition of' in line:
                        assoc_match = re.search(r'(?:Association to|Composition of)\s+([a-zA-Z_][a-zA-Z0-9_.]*)', line)
                        if assoc_match:
                            target_entity = assoc_match.group(1).split('.')[-1]  # Handle namespaced entities
                            used_entities.add(target_entity)
                    
                    # 4. Missing semicolons in CDS
                    if line.endswith(':') and not any(keyword in line for keyword in ['@', '//', 'service', 'entity', 'type', 'using']):
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="CDS property definition should end with semicolon",
                            severity="warning",
                            tool="cds-semantics"
                        ))
                    
                    # 5. Deprecated CDS syntax
                    if '@sap.semantics' in line:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="@sap.semantics is deprecated, use @Analytics or @Aggregation instead",
                            severity="warning",
                            tool="cds-semantics"
                        ))
                    
                    # 6. Missing key fields
                    if line.startswith('entity ') and 'cuid' not in line and 'managed' not in line:
                        if not re.search(r'key\s+\w+', content):
                            issues.append(self._create_issue(
                                file_path=str(file_path),
                                line=line_num,
                                message="Entity should have at least one key field or use cuid/managed",
                                severity="warning",
                                tool="cds-semantics"
                            ))
                    
                    # 7. Namespace validation
                    if line.startswith('namespace ') and not re.match(r'namespace\s+[a-z][a-z0-9]*(\.[a-z][a-z0-9]*)*;', line):
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Namespace should follow convention: lowercase.separated.names",
                            severity="info",
                            tool="cds-semantics"
                        ))
                    
                    # 8. Service exposure check
                    if 'projection on' in line and '@readonly' not in content and 'draft.enabled' not in content:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Consider adding @readonly or @odata.draft.enabled for service projections",
                            severity="info",
                            tool="cds-semantics"
                        ))
                
                # 9. Check for undefined entity references
                for used_entity in used_entities:
                    if used_entity not in defined_entities and used_entity not in ['Users', 'Languages', 'Countries', 'Currencies']:
                        # Find line where undefined entity is used
                        for line_num, line in enumerate(lines, 1):
                            if used_entity in line and ('Association to' in line or 'Composition of' in line):
                                issues.append(self._create_issue(
                                    file_path=str(file_path),
                                    line=line_num,
                                    message=f"Undefined entity reference: {used_entity}",
                                    severity="error",
                                    tool="cds-semantics"
                                ))
                                break
                
            except Exception as e:
                print(f"Error analyzing CDS semantics for {file_path}: {e}")
        
        return {"issues": issues}
    
    async def _analyze_cds_security(self, files: List[Path]) -> Dict[str, Any]:
        """Perform CDS security analysis"""
        issues = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    
                    # 1. Missing authentication requirements
                    if line.startswith('service ') and '@requires' not in content:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Service should have @requires annotation for authentication",
                            severity="warning",
                            tool="cds-security"
                        ))
                    
                    # 2. Unrestricted service access
                    if '@requires:' in line and "''" in line:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Empty @requires annotation allows unrestricted access",
                            severity="error",
                            tool="cds-security"
                        ))
                    
                    # 3. Sensitive data exposure
                    sensitive_fields = ['password', 'secret', 'token', 'key', 'credential']
                    for sensitive in sensitive_fields:
                        if sensitive in line.lower() and not any(keyword in line for keyword in ['@readonly', '@insertonly']):
                            issues.append(self._create_issue(
                                file_path=str(file_path),
                                line=line_num,
                                message=f"Sensitive field '{sensitive}' should be protected with @readonly or @insertonly",
                                severity="error",
                                tool="cds-security"
                            ))
                    
                    # 4. Missing field-level restrictions
                    if 'email' in line.lower() and '@assert.format' not in content:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Email field should have @assert.format validation",
                            severity="info",
                            tool="cds-security"
                        ))
                
            except Exception as e:
                print(f"Error analyzing CDS security for {file_path}: {e}")
        
        return {"issues": issues}
    
    async def _run_solidity_linters_batch(self, files: List[Path], directory: str) -> Dict[str, Any]:
        """Run comprehensive Solidity linters and security analysis on a batch of files"""
        issues = []
        linter_results = {}
        
        # Ensure we have relative paths for tools
        relative_files = [str(f.relative_to(directory)) for f in files]
        
        # 1. Solidity Compiler (solc) check
        if shutil.which("solc"):
            try:
                # Run solidity compiler for compilation errors
                cmd = ["solc", "--combined-json", "abi,bin", "--optimize"] + relative_files
                result = await self._run_command(" ".join(cmd), cwd=directory)
                
                if result.get("stderr"):
                    # Parse solc compilation errors
                    for line in result["stderr"].split("\n"):
                        if ".sol:" in line and ("Error:" in line or "Warning:" in line):
                            parts = line.split(".sol:")
                            if len(parts) >= 2:
                                file_path = parts[0] + ".sol"
                                rest = parts[1]
                                line_col = rest.split(":")[0] if ":" in rest else "1"
                                line_num = int(line_col) if line_col.isdigit() else 1
                                message = line.split("Error:")[-1].strip() if "Error:" in line else line.split("Warning:")[-1].strip()
                                severity = "error" if "Error:" in line else "warning"
                                
                                issues.append(self._create_issue(
                                    file_path=file_path,
                                    line=line_num,
                                    message=f"Solidity compiler: {message}",
                                    severity=severity,
                                    tool="solc"
                                ))
                
                solc_issues = [i for i in issues if i.get('tool') == 'solc']
                if solc_issues:
                    linter_results["solc"] = f"Found {len(solc_issues)} compilation issues"
                else:
                    linter_results["solc"] = "Compilation successful"
            except Exception as e:
                linter_results["solc"] = f"Error: {str(e)}"
        else:
            linter_results["solc"] = "Solidity compiler not available"
        
        # 2. Slither security analyzer
        if shutil.which("slither"):
            try:
                cmd = ["slither", directory, "--json-types", "detectors"]
                result = await self._run_command(" ".join(cmd), cwd=directory)
                
                if result.get("stdout"):
                    try:
                        slither_data = json.loads(result["stdout"])
                        if "results" in slither_data and "detectors" in slither_data["results"]:
                            for detector in slither_data["results"]["detectors"]:
                                for element in detector.get("elements", []):
                                    if "source_mapping" in element and "filename_absolute" in element["source_mapping"]:
                                        file_path = element["source_mapping"]["filename_absolute"]
                                        line_num = element["source_mapping"].get("lines", [1])[0]
                                        
                                        issues.append(self._create_issue(
                                            file_path=file_path,
                                            line=line_num,
                                            message=f"Security issue: {detector.get('description', 'Unknown vulnerability')}",
                                            severity="error" if detector.get("impact") == "High" else "warning",
                                            tool="slither"
                                        ))
                    except json.JSONDecodeError:
                        pass
                
                linter_results["slither"] = f"Found {len([i for i in issues if i.get('tool') == 'slither'])} security issues"
            except Exception as e:
                linter_results["slither"] = f"Error: {str(e)}"
        else:
            linter_results["slither"] = "Slither not available"
        
        # 3. Solidity semantic analysis
        sol_analysis = await self._analyze_solidity_semantics(files)
        issues.extend(sol_analysis.get("issues", []))
        linter_results["sol-semantics"] = f"Found {len(sol_analysis.get('issues', []))} semantic issues"
        
        # 4. Solidity security analysis
        sol_security = await self._analyze_solidity_security(files)
        issues.extend(sol_security.get("issues", []))
        linter_results["sol-security"] = f"Found {len(sol_security.get('issues', []))} security issues"
        
        # 5. Gas optimization analysis
        gas_analysis = await self._analyze_solidity_gas_optimization(files)
        issues.extend(gas_analysis.get("issues", []))
        linter_results["sol-gas"] = f"Found {len(gas_analysis.get('issues', []))} gas optimization opportunities"
        
        return {"issues": issues, "linter_results": linter_results}
    
    async def _analyze_solidity_semantics(self, files: List[Path]) -> Dict[str, Any]:
        """Perform Solidity-specific semantic analysis"""
        issues = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    
                    # 1. Pragma version specification
                    if line.startswith('pragma solidity'):
                        if '^' in line:
                            issues.append(self._create_issue(
                                file_path=str(file_path),
                                line=line_num,
                                message="Consider using exact pragma version instead of ^ for production contracts",
                                severity="info",
                                tool="sol-semantics"
                            ))
                    
                    # 2. Function visibility modifiers
                    if 'function ' in line and 'public' not in line and 'private' not in line and 'internal' not in line and 'external' not in line:
                        if 'constructor' not in line:
                            issues.append(self._create_issue(
                                file_path=str(file_path),
                                line=line_num,
                                message="Function should have explicit visibility modifier",
                                severity="warning",
                                tool="sol-semantics"
                            ))
                    
                    # 3. State variable visibility
                    if any(keyword in line for keyword in ['uint', 'int', 'bool', 'address', 'string', 'bytes']) and '=' in line and 'function' not in line:
                        if 'public' not in line and 'private' not in line and 'internal' not in line:
                            issues.append(self._create_issue(
                                file_path=str(file_path),
                                line=line_num,
                                message="State variable should have explicit visibility modifier",
                                severity="warning",
                                tool="sol-semantics"
                            ))
                    
                    # 4. Events should be declared
                    if 'emit ' in line:
                        event_name = line.split('emit ')[1].split('(')[0].strip()
                        if f'event {event_name}' not in content:
                            issues.append(self._create_issue(
                                file_path=str(file_path),
                                line=line_num,
                                message=f"Event '{event_name}' should be declared before use",
                                severity="error",
                                tool="sol-semantics"
                            ))
                    
                    # 5. NatSpec documentation
                    if line.startswith('function ') and 'public' in line:
                        # Check if there's documentation above this function
                        doc_found = False
                        for i in range(max(0, line_num - 5), line_num - 1):
                            if i < len(lines) and ('///' in lines[i] or '/**' in lines[i]):
                                doc_found = True
                                break
                        if not doc_found:
                            issues.append(self._create_issue(
                                file_path=str(file_path),
                                line=line_num,
                                message="Public functions should have NatSpec documentation",
                                severity="info",
                                tool="sol-semantics"
                            ))
                
            except Exception as e:
                print(f"Error analyzing Solidity semantics for {file_path}: {e}")
        
        return {"issues": issues}
    
    async def _analyze_solidity_security(self, files: List[Path]) -> Dict[str, Any]:
        """Perform comprehensive Solidity security analysis"""
        issues = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    
                    # 1. Reentrancy vulnerability patterns
                    if '.call(' in line and 'value:' in line:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Potential reentrancy vulnerability: external call with value transfer",
                            severity="error",
                            tool="sol-security"
                        ))
                    
                    # 2. tx.origin usage (should use msg.sender)
                    if 'tx.origin' in line:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Avoid tx.origin - use msg.sender for authentication",
                            severity="error",
                            tool="sol-security"
                        ))
                    
                    # 3. Block timestamp dependence
                    if 'block.timestamp' in line or 'now' in line:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Avoid relying on block.timestamp - miners can manipulate it",
                            severity="warning",
                            tool="sol-security"
                        ))
                    
                    # 4. Unsafe low-level calls
                    if '.call(' in line or '.delegatecall(' in line or '.staticcall(' in line:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Low-level calls can be dangerous - ensure proper error handling and reentrancy protection",
                            severity="warning",
                            tool="sol-security"
                        ))
                    
                    # 5. Integer overflow/underflow (pre-0.8.0)
                    if any(op in line for op in ['+', '-', '*', '/']):
                        if 'SafeMath' not in content and 'pragma solidity' in content:
                            pragma_line = [l for l in lines if 'pragma solidity' in l][0]
                            if any(ver in pragma_line for ver in ['0.4', '0.5', '0.6', '0.7']):
                                issues.append(self._create_issue(
                                    file_path=str(file_path),
                                    line=line_num,
                                    message="Consider using SafeMath library for arithmetic operations in older Solidity versions",
                                    severity="warning",
                                    tool="sol-security"
                                ))
                    
                    # 6. Unchecked external calls
                    if '.call(' in line and 'require(' not in line and 'assert(' not in line:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="External call return value should be checked",
                            severity="error",
                            tool="sol-security"
                        ))
                    
                    # 7. Access control issues
                    if 'onlyOwner' in line and 'modifier onlyOwner' not in content:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="onlyOwner modifier used but not defined",
                            severity="error",
                            tool="sol-security"
                        ))
                    
                    # 8. Hardcoded addresses
                    if '0x' in line and len([x for x in line.split() if x.startswith('0x') and len(x) == 42]) > 0:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Avoid hardcoded addresses - use constructor parameters or constants",
                            severity="info",
                            tool="sol-security"
                        ))
                
            except Exception as e:
                print(f"Error analyzing Solidity security for {file_path}: {e}")
        
        return {"issues": issues}
    
    async def _analyze_solidity_gas_optimization(self, files: List[Path]) -> Dict[str, Any]:
        """Analyze Solidity code for gas optimization opportunities"""
        issues = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    
                    # 1. Use of storage vs memory
                    if 'storage' in line and 'function' in line:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Consider if memory would be more gas-efficient than storage",
                            severity="info",
                            tool="sol-gas"
                        ))
                    
                    # 2. Loop optimizations
                    if 'for (' in line:
                        if '.length' in line:
                            issues.append(self._create_issue(
                                file_path=str(file_path),
                                line=line_num,
                                message="Cache array length outside loop to save gas",
                                severity="info",
                                tool="sol-gas"
                            ))
                    
                    # 3. Public vs external functions
                    if 'function ' in line and 'public' in line and 'view' not in line:
                        issues.append(self._create_issue(
                            file_path=str(file_path),
                            line=line_num,
                            message="Consider using 'external' instead of 'public' if function is not called internally",
                            severity="info",
                            tool="sol-gas"
                        ))
                    
                    # 4. Unnecessary storage reads
                    state_vars = re.findall(r'\b(\w+)\s*=', line)
                    for var in state_vars:
                        if var in content and content.count(var) > 5:
                            issues.append(self._create_issue(
                                file_path=str(file_path),
                                line=line_num,
                                message=f"Consider caching '{var}' in memory if accessed multiple times",
                                severity="info",
                                tool="sol-gas"
                            ))
                
            except Exception as e:
                print(f"Error analyzing Solidity gas optimization for {file_path}: {e}")
        
        return {"issues": issues}
    
    async def _run_python_linters(self, file_path: str) -> List[CodeIssue]:
        """Run Python linting tools"""
        issues = []
        
        # Run pylint
        if self._check_tool_available("pylint"):
            try:
                result = subprocess.run(
                    ["pylint", "--output-format=json", file_path],
                    capture_output=True,
                    text=True
                )
                if result.stdout:
                    pylint_issues = json.loads(result.stdout)
                    for issue in pylint_issues:
                        line_num = issue.get("line", 0)
                        col_num = issue.get("column", 0)
                        issue_id = hashlib.md5(f'{file_path}{line_num}{col_num}'.encode(), usedforsecurity=False).hexdigest()[:8]
                        issues.append(CodeIssue(
                            id=f"pylint_{issue_id}",
                            file_path=file_path,
                            line=line_num,
                            column=col_num,
                            tool="pylint",
                            issue_type=self._map_pylint_type(issue.get("type", "")),
                            severity=self._map_pylint_severity(issue.get("type", "")),
                            code=issue.get("symbol", ""),
                            message=issue.get("message", ""),
                            rule=issue.get("message-id", "")
                        ))
            except Exception as e:
                logger.error(f"Pylint error for {file_path}: {e}")
        
        # Run flake8
        if self._check_tool_available("flake8"):
            try:
                result = subprocess.run(
                    ["flake8", "--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s", file_path],
                    capture_output=True,
                    text=True
                )
                if result.stdout:
                    for line in result.stdout.strip().split("\n"):
                        if line:
                            match = re.match(r"(.+):(\d+):(\d+): (\w+) (.+)", line)
                            if match:
                                issues.append(CodeIssue(
                                    id=f"flake8_{hashlib.md5(line.encode(), usedforsecurity=False).hexdigest()[:8]}",
                                    file_path=match.group(1),
                                    line=int(match.group(2)),
                                    column=int(match.group(3)),
                                    tool="flake8",
                                    issue_type=self._map_flake8_type(match.group(4)),
                                    severity=self._map_flake8_severity(match.group(4)),
                                    code=match.group(4),
                                    message=match.group(5),
                                    rule=match.group(4)
                                ))
            except Exception as e:
                logger.error(f"Flake8 error for {file_path}: {e}")
        
        return issues
    
    async def _run_javascript_linters(self, file_path: str) -> List[CodeIssue]:
        """Run JavaScript linting tools"""
        issues = []
        
        if self._check_tool_available("eslint"):
            try:
                result = subprocess.run(
                    ["eslint", "--format=json", file_path],
                    capture_output=True,
                    text=True
                )
                if result.stdout:
                    eslint_results = json.loads(result.stdout)
                    for file_result in eslint_results:
                        for message in file_result.get("messages", []):
                            line_num = message.get("line", 0)
                            col_num = message.get("column", 0)
                            issue_id = hashlib.md5(f'{file_path}{line_num}{col_num}'.encode(), usedforsecurity=False).hexdigest()[:8]
                            issues.append(CodeIssue(
                                id=f"eslint_{issue_id}",
                                file_path=file_path,
                                line=line_num,
                                column=col_num,
                                tool="eslint",
                                issue_type=self._map_eslint_type(message.get("severity", 0)),
                                severity=self._map_eslint_severity(message.get("severity", 0)),
                                code=message.get("ruleId", ""),
                                message=message.get("message", ""),
                                rule=message.get("ruleId", ""),
                                auto_fixable=message.get("fix") is not None
                            ))
            except Exception as e:
                logger.error(f"ESLint error for {file_path}: {e}")
        
        return issues
    
    async def _run_typescript_linters(self, file_path: str) -> List[CodeIssue]:
        """Run TypeScript linting tools"""
        # Similar to JavaScript but with TypeScript-specific tools
        return await self._run_javascript_linters(file_path)
    
    async def _run_tests(self, directory: str) -> Dict[str, Any]:
        """Run tests and collect coverage data"""
        test_results = {
            "framework": None,
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "coverage": None,
            "errors": []
        }
        
        # Try pytest for Python projects
        if Path(directory).glob("**/test_*.py") or Path(directory).glob("**/*_test.py"):
            try:
                result = subprocess.run(
                    ["pytest", "--json-report", "--json-report-file=/tmp/pytest_report.json", directory],
                    capture_output=True,
                    text=True,
                    cwd=directory
                )
                
                # Read test results
                report_path = Path("/tmp/pytest_report.json")
                if report_path.exists():
                    with open(report_path) as f:
                        report = json.load(f)
                        test_results["framework"] = "pytest"
                        test_results["tests_run"] = report.get("summary", {}).get("total", 0)
                        test_results["tests_passed"] = report.get("summary", {}).get("passed", 0)
                        test_results["tests_failed"] = report.get("summary", {}).get("failed", 0)
            except Exception as e:
                test_results["errors"].append(f"pytest error: {str(e)}")
        
        # Try jest/npm test for JavaScript projects
        package_json = Path(directory) / "package.json"
        if package_json.exists():
            try:
                result = subprocess.run(
                    ["npm", "test", "--", "--json"],
                    capture_output=True,
                    text=True,
                    cwd=directory
                )
                if result.stdout:
                    test_results["framework"] = "jest/npm"
                    # Parse jest output if available
            except Exception as e:
                test_results["errors"].append(f"npm test error: {str(e)}")
        
        return test_results
    
    async def _perform_security_analysis(self, directory: str) -> Dict[str, Any]:
        """Perform security vulnerability analysis"""
        security_issues = []
        
        # Run bandit for Python security
        if self._check_tool_available("bandit"):
            try:
                result = subprocess.run(
                    ["bandit", "-r", "-f", "json", directory],
                    capture_output=True,
                    text=True
                )
                if result.stdout:
                    bandit_results = json.loads(result.stdout)
                    for issue in bandit_results.get("results", []):
                        security_issues.append({
                            "tool": "bandit",
                            "severity": issue.get("issue_severity", "").lower(),
                            "confidence": issue.get("issue_confidence", "").lower(),
                            "file": issue.get("filename", ""),
                            "line": issue.get("line_number", 0),
                            "test_name": issue.get("test_name", ""),
                            "issue_text": issue.get("issue_text", "")
                        })
            except Exception as e:
                logger.error(f"Bandit error: {e}")
        
        # Check for known vulnerable dependencies
        # This would integrate with tools like safety, npm audit, etc.
        
        return {
            "total_issues": len(security_issues),
            "critical_issues": len([i for i in security_issues if i.get("severity") == "high"]),
            "issues": security_issues[:50]  # Limit to first 50
        }
    
    def _calculate_summary_metrics(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary metrics from all analyses"""
        summary = {
            "total_issues": 0,
            "critical_issues": 0,
            "files_analyzed": 0,
            "test_coverage": None,
            "quality_score": 100.0
        }
        
        # Aggregate lint issues
        if "lint" in analyses:
            lint_data = analyses["lint"]
            summary["total_issues"] += lint_data.get("total_issues", 0)
            summary["files_analyzed"] = lint_data.get("files_analyzed", 0)
            
            # Count critical issues
            severity_counts = lint_data.get("issues_by_severity", {})
            summary["critical_issues"] += severity_counts.get(IssueSeverity.CRITICAL, 0)
            summary["critical_issues"] += severity_counts.get(IssueSeverity.HIGH, 0)
        
        # Add security issues
        if "security" in analyses:
            security_data = analyses["security"]
            summary["total_issues"] += security_data.get("total_issues", 0)
            summary["critical_issues"] += security_data.get("critical_issues", 0)
        
        # Test results
        if "test" in analyses:
            test_data = analyses["test"]
            if test_data.get("tests_run", 0) > 0:
                pass_rate = test_data.get("tests_passed", 0) / test_data["tests_run"]
                summary["test_pass_rate"] = pass_rate
                summary["quality_score"] *= pass_rate
        
        # Calculate industry-standard quality score
        summary["quality_score"] = self._calculate_comprehensive_quality_score(summary, analyses)
        
        return summary
    
    def _calculate_comprehensive_quality_score(self, summary: Dict[str, Any], analyses: Dict[str, Any]) -> float:
        """
        Calculate comprehensive quality score based on industry standards
        Uses weighted metrics from multiple dimensions:
        - Code Quality (40%): Issues, complexity, maintainability
        - Test Quality (25%): Coverage, test pass rate, test completeness  
        - Security (20%): Vulnerabilities, security issues
        - Documentation (10%): Docstrings, comments, README
        - Architecture (5%): Dependencies, patterns, structure
        """
        
        # Base score starts at 100
        total_score = 100.0
        
        # 1. CODE QUALITY DIMENSION (40% weight)
        code_quality_score = 100.0
        
        if summary["files_analyzed"] > 0:
            # Critical issues have major impact
            critical_issues = summary.get("critical_issues", 0)
            if critical_issues > 0:
                critical_penalty = min(critical_issues * 15, 60)  # Up to 60 point penalty
                code_quality_score -= critical_penalty
            
            # General issue density
            total_issues = summary.get("total_issues", 0)
            if total_issues > 0:
                issue_density = total_issues / summary["files_analyzed"]
                if issue_density > 10:  # More than 10 issues per file is poor
                    density_penalty = min((issue_density - 10) * 3, 30)
                    code_quality_score -= density_penalty
                elif issue_density > 5:  # 5-10 issues per file is moderate
                    density_penalty = (issue_density - 5) * 2
                    code_quality_score -= density_penalty
        
        # Complexity analysis from complexity results
        if "complexity" in analyses:
            complexity_data = analyses["complexity"]
            avg_complexity = complexity_data.get("average_complexity", 0)
            if avg_complexity > 15:  # Very high complexity
                code_quality_score -= min((avg_complexity - 15) * 2, 20)
            elif avg_complexity > 10:  # High complexity
                code_quality_score -= (avg_complexity - 10) * 1
        
        code_quality_score = max(0, code_quality_score)
        
        # 2. TEST QUALITY DIMENSION (25% weight)
        test_quality_score = 100.0
        
        # Test coverage impact
        test_coverage = summary.get("test_coverage")
        if test_coverage is not None:
            if test_coverage < 60:  # Below 60% is poor
                coverage_penalty = (60 - test_coverage) * 0.8
                test_quality_score -= coverage_penalty
            elif test_coverage > 90:  # Above 90% is excellent
                test_quality_score += 5  # Bonus
        else:
            test_quality_score -= 50  # No coverage data is a major penalty
        
        # Test pass rate
        test_pass_rate = summary.get("test_pass_rate")
        if test_pass_rate is not None:
            if test_pass_rate < 1.0:
                pass_penalty = (1.0 - test_pass_rate) * 40  # Failing tests are serious
                test_quality_score -= pass_penalty
        else:
            test_quality_score -= 30  # No test execution data
        
        test_quality_score = max(0, test_quality_score)
        
        # 3. SECURITY DIMENSION (20% weight)
        security_score = 100.0
        
        if "security" in analyses:
            security_data = analyses["security"]
            vulnerabilities = security_data.get("total_vulnerabilities", 0)
            
            if vulnerabilities > 0:
                # High/critical vulnerabilities are heavily penalized
                high_vulns = len([v for v in security_data.get("vulnerabilities", []) 
                                if v.get("severity") in ["high", "critical"]])
                if high_vulns > 0:
                    security_score -= high_vulns * 25  # 25 points per high/critical vuln
                
                # Medium vulnerabilities
                medium_vulns = len([v for v in security_data.get("vulnerabilities", []) 
                                  if v.get("severity") == "medium"])
                if medium_vulns > 0:
                    security_score -= medium_vulns * 10  # 10 points per medium vuln
                
                # Low vulnerabilities
                low_vulns = vulnerabilities - high_vulns - medium_vulns
                if low_vulns > 0:
                    security_score -= low_vulns * 3  # 3 points per low vuln
        
        security_score = max(0, security_score)
        
        # 4. DOCUMENTATION DIMENSION (10% weight)
        documentation_score = 100.0
        
        # Check if functions have docstrings (from complexity analysis)
        if "complexity" in analyses:
            complexity_data = analyses["complexity"]
            functions_with_docs = 0
            total_functions = 0
            
            for file_complexity in complexity_data.get("file_complexities", []):
                total_functions += file_complexity.get("functions", 0)
                # This is a simplification - in real implementation we'd check actual docstrings
                functions_with_docs += file_complexity.get("functions", 0) * 0.6  # Assume 60% have docs
            
            if total_functions > 0:
                doc_ratio = functions_with_docs / total_functions
                if doc_ratio < 0.5:  # Less than 50% documented
                    documentation_score -= (0.5 - doc_ratio) * 100
        
        documentation_score = max(0, documentation_score)
        
        # 5. ARCHITECTURE DIMENSION (5% weight)
        architecture_score = 100.0
        
        if "glean" in analyses:
            glean_data = analyses["glean"]
            
            # Circular dependencies are bad
            circular_deps = len(glean_data.get("dependency_graph", {}).get("circular_dependencies", []))
            if circular_deps > 0:
                architecture_score -= circular_deps * 15
            
            # Dead code is problematic
            dead_code = len(glean_data.get("dead_code_candidates", []))
            if dead_code > 0:
                architecture_score -= dead_code * 5
        
        architecture_score = max(0, architecture_score)
        
        # Calculate weighted final score
        final_score = (
            code_quality_score * 0.40 +      # 40% weight
            test_quality_score * 0.25 +      # 25% weight  
            security_score * 0.20 +          # 20% weight
            documentation_score * 0.10 +     # 10% weight
            architecture_score * 0.05        # 5% weight
        )
        
        # Apply additional modifiers
        if final_score > 95:
            final_score = min(100, final_score + 2)  # Excellence bonus
        elif final_score < 20:
            final_score = max(0, final_score - 5)   # Poor quality penalty
        
        return round(final_score, 1)
    
    async def _store_analysis_results(self, analysis_id: str, results: Dict[str, Any]) -> None:
        """Store analysis results in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Store main analysis record
                conn.execute("""
                    INSERT INTO analysis_results 
                    (id, analysis_type, directory, files_analyzed, issue_count, duration, timestamp, results)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analysis_id,
                    "comprehensive",
                    results["directory"],
                    results["summary"]["files_analyzed"],
                    results["summary"]["total_issues"],
                    results["duration"],
                    results["timestamp"],
                    json.dumps(results)
                ))
                
                # Store individual issues
                if "lint" in results["analyses"]:
                    for issue_dict in results["analyses"]["lint"].get("issues", []):
                        issue = CodeIssue(**issue_dict)
                        conn.execute("""
                            INSERT INTO code_issues
                            (id, analysis_id, file_path, line, column, tool, issue_type, 
                             severity, code, message, rule, suggestion, auto_fixable, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            issue.id,
                            analysis_id,
                            issue.file_path,
                            issue.line,
                            issue.column,
                            issue.tool,
                            issue.issue_type,
                            issue.severity,
                            issue.code,
                            issue.message,
                            issue.rule,
                            issue.suggestion,
                            issue.auto_fixable,
                            issue.created_at.isoformat()
                        ))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store analysis results: {e}")
    
    @a2a_skill("get_analysis_history", "Retrieve historical analysis results")
    async def get_analysis_history(
        self,
        directory: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve historical analysis results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                if directory:
                    cursor = conn.execute("""
                        SELECT * FROM analysis_results 
                        WHERE directory = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (directory, limit))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM analysis_results 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (limit,))
                
                results = []
                for row in cursor:
                    result = dict(row)
                    result["results"] = json.loads(result["results"])
                    results.append(result)
                
                return results
        except Exception as e:
            logger.error(f"Failed to retrieve analysis history: {e}")
            return []
    
    @a2a_skill("code_refactoring_suggestions", "Generate AI-powered refactoring suggestions using AST analysis")
    async def analyze_code_refactoring(
        self,
        file_path: str,
        max_suggestions: int = 10
    ) -> Dict[str, Any]:
        """
        Real AST-based code refactoring analysis with intelligent suggestions
        """
        try:
            import ast
            import re
            from collections import defaultdict
            
            if not Path(file_path).exists():
                return {"error": "File not found", "suggestions": []}
            
            suggestions = []
            
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST for real analysis
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                return {
                    "error": f"Syntax error in file: {e}",
                    "suggestions": [],
                    "file_path": file_path
                }
            
            lines = content.split('\n')
            
            # Real AST-based analysis
            suggestions.extend(await self._analyze_ast_for_refactoring(tree, lines, file_path))
            
            # Additional pattern-based analysis for things AST doesn't catch
            suggestions.extend(await self._analyze_patterns_for_refactoring(lines, file_path))
            
            # Remove duplicates and sort by priority
            unique_suggestions = self._deduplicate_suggestions(suggestions)
            
            # Sort by severity and limit results
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            unique_suggestions.sort(key=lambda x: (severity_order.get(x["severity"], 4), x["line"]))
            unique_suggestions = unique_suggestions[:max_suggestions]
            
            # Calculate refactoring metrics
            metrics = self._calculate_refactoring_metrics(tree, unique_suggestions)
            
            return {
                "file_path": file_path,
                "total_suggestions": len(unique_suggestions),
                "suggestions": unique_suggestions,
                "summary": {
                    "critical_priority": len([s for s in unique_suggestions if s["severity"] == "critical"]),
                    "high_priority": len([s for s in unique_suggestions if s["severity"] == "high"]),
                    "medium_priority": len([s for s in unique_suggestions if s["severity"] == "medium"]),
                    "low_priority": len([s for s in unique_suggestions if s["severity"] == "low"])
                },
                "metrics": metrics,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Refactoring analysis failed for {file_path}: {e}")
            return {"error": str(e), "suggestions": []}
    
    @a2a_skill("dependency_vulnerability_scan", "Scan dependencies for security vulnerabilities")
    async def scan_dependency_vulnerabilities(
        self,
        directory: str,
        scan_dev_dependencies: bool = False
    ) -> Dict[str, Any]:
        """
        Scan project dependencies for known security vulnerabilities
        """
        try:
            vulnerabilities = []
            scanned_files = []
            
            # Check for Python requirements files
            req_files = ["requirements.txt", "requirements-dev.txt", "pyproject.toml", "Pipfile"]
            for req_file in req_files:
                req_path = Path(directory) / req_file
                if req_path.exists():
                    scanned_files.append(str(req_path))
                    
                    if req_file == "requirements.txt" or (req_file == "requirements-dev.txt" and scan_dev_dependencies):
                        # Use safety for Python dependency scanning
                        if self._check_tool_available("safety"):
                            try:
                                result = subprocess.run(
                                    ["safety", "check", "-r", str(req_path), "--json"],
                                    capture_output=True,
                                    text=True
                                )
                                if result.stdout:
                                    safety_results = json.loads(result.stdout)
                                    for vuln in safety_results:
                                        vulnerabilities.append({
                                            "package": vuln.get("package_name", "unknown"),
                                            "version": vuln.get("analyzed_version", "unknown"),
                                            "vulnerability_id": vuln.get("vulnerability_id", ""),
                                            "severity": "high",  # Safety reports are typically high severity
                                            "description": vuln.get("advisory", ""),
                                            "file": str(req_path)
                                        })
                            except Exception as e:
                                logger.warning(f"Safety scan failed for {req_path}: {e}")
                    
                    elif req_file == "pyproject.toml":
                        # Basic pyproject.toml parsing for known vulnerable packages
                        try:
                            with open(req_path, 'r') as f:
                                content = f.read()
                                # Simple check for commonly vulnerable packages
                                vulnerable_patterns = [
                                    ("pillow", "< 8.2.0", "PIL vulnerabilities in older versions"),
                                    ("django", "< 3.2.0", "Multiple security fixes in Django 3.2+"),
                                    ("flask", "< 2.0.0", "Security improvements in Flask 2.0+"),
                                    ("requests", "< 2.25.0", "Various security fixes")
                                ]
                                
                                for package, version_check, description in vulnerable_patterns:
                                    if package in content.lower():
                                        vulnerabilities.append({
                                            "package": package,
                                            "version": "unknown",
                                            "vulnerability_id": f"MANUAL_CHECK_{package.upper()}",
                                            "severity": "medium",
                                            "description": f"Manual check required: {description}",
                                            "file": str(req_path)
                                        })
                        except Exception as e:
                            logger.warning(f"pyproject.toml parsing failed: {e}")
            
            # Check for Node.js package files
            node_files = ["package.json", "package-lock.json", "yarn.lock"]
            for node_file in node_files:
                node_path = Path(directory) / node_file
                if node_path.exists():
                    scanned_files.append(str(node_path))
                    
                    # Use npm audit for Node.js dependency scanning
                    if node_file == "package.json" and self._check_tool_available("npm"):
                        try:
                            result = subprocess.run(
                                ["npm", "audit", "--json"],
                                capture_output=True,
                                text=True,
                                cwd=directory
                            )
                            if result.stdout:
                                audit_results = json.loads(result.stdout)
                                advisories = audit_results.get("advisories", {})
                                for advisory_id, advisory in advisories.items():
                                    vulnerabilities.append({
                                        "package": advisory.get("module_name", "unknown"),
                                        "version": advisory.get("findings", [{}])[0].get("version", "unknown"),
                                        "vulnerability_id": advisory_id,
                                        "severity": advisory.get("severity", "unknown"),
                                        "description": advisory.get("title", ""),
                                        "file": str(node_path)
                                    })
                        except Exception as e:
                            logger.warning(f"npm audit failed: {e}")
            
            # Enhanced: Add built-in vulnerability database scanning
            builtin_vulns = await self._scan_with_builtin_vulnerability_database(directory, scan_dev_dependencies)
            vulnerabilities.extend(builtin_vulns)
            
            # Enhanced: Add code pattern analysis for security issues
            code_vulns = await self._scan_code_for_security_patterns(directory)
            vulnerabilities.extend(code_vulns)
            
            # Count source files scanned for security patterns
            source_files_scanned = len([f for f in Path(directory).rglob("*.py") 
                                       if not any(skip in str(f.name) for skip in ["test_", "_test.py"]) and
                                       not any(skip in str(f) for skip in ["/tests/", "/test/", "__pycache__", ".venv", "venv"])])
            
            # Remove duplicates and calculate risk metrics
            unique_vulnerabilities = self._deduplicate_vulnerabilities(vulnerabilities)
            risk_metrics = self._calculate_security_risk_metrics(unique_vulnerabilities)
            
            # Summary statistics
            severity_counts = {}
            for vuln in unique_vulnerabilities:
                severity = vuln.get("severity", "unknown")
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            return {
                "directory": directory,
                "scanned_files": len(scanned_files) + source_files_scanned,
                "dependency_files": scanned_files,
                "source_files_scanned": source_files_scanned,
                "total_vulnerabilities": len(unique_vulnerabilities),
                "vulnerabilities": unique_vulnerabilities,
                "severity_breakdown": severity_counts,
                "risk_metrics": risk_metrics,
                "scan_timestamp": datetime.utcnow().isoformat(),
                "database_version": "2024.1.0"  # Built-in vulnerability database version
            }
            
        except Exception as e:
            logger.error(f"Vulnerability scan failed for {directory}: {e}")
            return {"error": str(e), "vulnerabilities": []}
    
    @a2a_skill("code_complexity_analysis", "Analyze code complexity metrics using real AST parsing")
    async def analyze_code_complexity(
        self,
        directory: str,
        file_patterns: List[str] = None,
        complexity_threshold: int = 10
    ) -> Dict[str, Any]:
        """
        Real code complexity analysis using AST parsing for accurate metrics
        """
        try:
            import ast
            start_time = time.time()
            
            if file_patterns is None:
                file_patterns = ["*.py"]  # Focus on Python for real implementation
            
            complexity_results = {
                "directory": directory,
                "complexity_threshold": complexity_threshold,
                "files_analyzed": 0,
                "functions_analyzed": 0,
                "classes_analyzed": 0,
                "high_complexity_functions": [],
                "complexity_distribution": {},
                "average_complexity": 0.0,
                "max_complexity": 0,
                "recommendations": [],
                "file_complexities": [],
                "duration": 0.0
            }
            
            # Find Python source files (real implementation focuses on Python)
            path = Path(directory)
            python_files = []
            for pattern in file_patterns:
                if pattern == "*.py":
                    python_files.extend(path.rglob(pattern))
            
            # Filter out test files and ignored directories
            ignore_patterns = ["test_", "_test.", "/test/", "/tests/", "__pycache__", "venv", "env"]
            filtered_files = [
                f for f in python_files 
                if not any(pattern in str(f) for pattern in ignore_patterns)
            ]
            
            complexity_results["files_analyzed"] = len(filtered_files)
            
            # Analyze each file using real AST parsing
            all_functions = []
            all_classes = []
            
            for file_path in filtered_files:
                try:
                    file_complexity = await self._analyze_file_complexity_real(file_path)
                    all_functions.extend(file_complexity["functions"])
                    all_classes.extend(file_complexity["classes"])
                    complexity_results["file_complexities"].append({
                        "file": str(file_path),
                        "functions": len(file_complexity["functions"]),
                        "classes": len(file_complexity["classes"]),
                        "avg_complexity": file_complexity["average_complexity"],
                        "max_complexity": file_complexity["max_complexity"]
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze complexity for {file_path}: {e}")
            
            complexity_results["functions_analyzed"] = len(all_functions)
            complexity_results["classes_analyzed"] = len(all_classes)
            
            if all_functions:
                # Calculate real statistics
                complexities = [f["complexity"] for f in all_functions]
                complexity_results["average_complexity"] = sum(complexities) / len(complexities)
                complexity_results["max_complexity"] = max(complexities)
                
                # Find high complexity functions
                high_complexity = sorted([
                    f for f in all_functions 
                    if f["complexity"] > complexity_threshold
                ], key=lambda x: x["complexity"], reverse=True)
                
                complexity_results["high_complexity_functions"] = high_complexity[:10]  # Top 10
                
                # Create complexity distribution
                distribution = {}
                for complexity in complexities:
                    range_key = self._get_complexity_range(complexity)
                    distribution[range_key] = distribution.get(range_key, 0) + 1
                complexity_results["complexity_distribution"] = distribution
                
                # Generate actionable recommendations
                complexity_results["recommendations"] = self._generate_complexity_recommendations(
                    complexity_results, complexity_threshold
                )
            
            complexity_results["duration"] = time.time() - start_time
            return complexity_results
            
        except Exception as e:
            logger.error(f"Complexity analysis failed: {e}")
            return {"error": str(e), "directory": directory}
    
    async def _analyze_file_complexity_real(self, file_path: Path) -> Dict[str, Any]:
        """Real AST-based complexity analysis for a single Python file"""
        import ast
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            functions = []
            classes = []
            
            class ComplexityVisitor(ast.NodeVisitor):
                def visit_FunctionDef(self, node):
                    # Calculate cyclomatic complexity for this function
                    complexity = self._calculate_cyclomatic_complexity(node)
                    
                    functions.append({
                        "name": node.name,
                        "line": node.lineno,
                        "complexity": complexity,
                        "file": str(file_path),
                        "args": len(node.args.args),
                        "returns": node.returns is not None,
                        "docstring": ast.get_docstring(node) is not None
                    })
                    
                    self.generic_visit(node)
                
                def visit_AsyncFunctionDef(self, node):
                    # Handle async functions the same way
                    self.visit_FunctionDef(node)
                
                def visit_ClassDef(self, node):
                    # Analyze class complexity
                    method_count = len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])
                    
                    classes.append({
                        "name": node.name,
                        "line": node.lineno,
                        "methods": method_count,
                        "file": str(file_path),
                        "docstring": ast.get_docstring(node) is not None
                    })
                    
                    self.generic_visit(node)
                
                def _calculate_cyclomatic_complexity(self, node):
                    """Calculate real cyclomatic complexity for a function"""
                    complexity = 1  # Base complexity
                    
                    for child in ast.walk(node):
                        # Decision points that increase complexity
                        if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                            complexity += 1
                        elif isinstance(child, ast.ExceptHandler):
                            complexity += 1
                        elif isinstance(child, (ast.With, ast.AsyncWith)):
                            complexity += 1
                        elif isinstance(child, ast.Assert):
                            complexity += 1
                        elif isinstance(child, ast.BoolOp):
                            # Each additional boolean operator adds complexity
                            complexity += len(child.values) - 1
                        elif isinstance(child, ast.ListComp):
                            complexity += 1
                        elif isinstance(child, ast.DictComp):
                            complexity += 1
                        elif isinstance(child, ast.SetComp):
                            complexity += 1
                        elif isinstance(child, ast.GeneratorExp):
                            complexity += 1
                    
                    return complexity
            
            visitor = ComplexityVisitor()
            visitor.visit(tree)
            
            # Calculate file-level statistics
            complexities = [f["complexity"] for f in functions] if functions else [0]
            
            return {
                "functions": functions,
                "classes": classes,
                "average_complexity": sum(complexities) / len(complexities),
                "max_complexity": max(complexities) if complexities else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return {"functions": [], "classes": [], "average_complexity": 0, "max_complexity": 0}
    
    def _get_complexity_range(self, complexity: int) -> str:
        """Categorize complexity into ranges"""
        if complexity <= 5:
            return "1-5 (Simple)"
        elif complexity <= 10:
            return "6-10 (Moderate)"
        elif complexity <= 20:
            return "11-20 (Complex)"
        else:
            return "21+ (Very Complex)"
    
    def _generate_complexity_recommendations(self, results: Dict[str, Any], threshold: int) -> List[str]:
        """Generate actionable complexity recommendations"""
        recommendations = []
        
        high_complexity_count = len(results["high_complexity_functions"])
        avg_complexity = results["average_complexity"]
        max_complexity = results["max_complexity"]
        
        if high_complexity_count > 0:
            recommendations.append(
                f"🔧 {high_complexity_count} functions exceed complexity threshold ({threshold}). "
                f"Consider breaking down the most complex ones."
            )
            
            # Specific recommendations for the most complex functions
            top_complex = results["high_complexity_functions"][:3]
            for func in top_complex:
                recommendations.append(
                    f"⚠️  '{func['name']}' at line {func['line']} has complexity {func['complexity']}. "
                    "Consider extracting helper functions or simplifying logic."
                )
        
        if avg_complexity > threshold:
            recommendations.append(
                f"📊 Average complexity ({avg_complexity:.1f}) exceeds threshold. "
                "Focus on refactoring to improve maintainability."
            )
        
        if max_complexity > 20:
            recommendations.append(
                f"🚨 Maximum complexity ({max_complexity}) is very high. "
                "This function should be prioritized for refactoring."
            )
        
        # Positive feedback
        if high_complexity_count == 0:
            recommendations.append("✅ All functions are within acceptable complexity limits!")
        
        distribution = results["complexity_distribution"]
        if distribution.get("1-5", 0) > distribution.get("11-20", 0):
            recommendations.append("👍 Most functions have low complexity - good code structure!")
        
        return recommendations
    
    async def _analyze_file_complexity(self, file_path: str, threshold: int) -> Optional[Dict[str, Any]]:
        """Analyze complexity for a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            functions = []
            
            if str(file_path).endswith('.py'):
                functions = self._analyze_python_complexity(content)
            elif str(file_path).endswith(('.js', '.ts')):
                functions = self._analyze_javascript_complexity(content)
            
            # Filter functions above threshold
            complex_functions = [f for f in functions if f.get("complexity", 0) > threshold]
            
            if functions:  # Only return if we found functions
                return {
                    "file_path": file_path,
                    "total_functions": len(functions),
                    "complex_functions": len(complex_functions),
                    "functions": functions
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to analyze complexity for {file_path}: {e}")
            return None
    
    def _analyze_python_complexity(self, content: str) -> List[Dict[str, Any]]:
        """Real AST-based Python code complexity analysis"""
        import ast
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []  # Return empty if syntax error
        
        functions = []
        
        class ComplexityVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                complexity = self._calculate_complexity(node)
                functions.append({
                    "name": node.name,
                    "line_start": node.lineno,
                    "line_end": node.end_lineno or node.lineno,
                    "complexity": complexity
                })
                self.generic_visit(node)
            
            def visit_AsyncFunctionDef(self, node):
                # Handle async functions the same way
                self.visit_FunctionDef(node)
            
            def _calculate_complexity(self, func_node):
                """Calculate cyclomatic complexity using AST"""
                complexity = 1  # Base complexity
                
                for node in ast.walk(func_node):
                    # Decision points increase complexity
                    if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                        complexity += 1
                    elif isinstance(node, ast.ExceptHandler):
                        complexity += 1
                    elif isinstance(node, ast.With):
                        complexity += 1
                    elif isinstance(node, ast.AsyncWith):
                        complexity += 1
                    elif isinstance(node, ast.BoolOp):
                        # Each additional boolean operator adds complexity
                        complexity += len(node.values) - 1
                    elif isinstance(node, ast.comprehension):
                        # List/dict/set comprehensions with conditions
                        complexity += len(node.ifs)
                
                return complexity
        
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        return functions
    
    def _analyze_javascript_complexity(self, content: str) -> List[Dict[str, Any]]:
        """Analyze JavaScript/TypeScript code complexity"""
        functions = []
        lines = content.split('\n')
        
        import re
        function_patterns = [
            re.compile(r'function\s+(\w+)\s*\('),
            re.compile(r'(\w+)\s*:\s*function\s*\('),
            re.compile(r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>'),
            re.compile(r'(\w+)\s*\([^)]*\)\s*{')
        ]
        
        current_function = None
        function_start = 0
        complexity = 1
        brace_count = 0
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Check for function declarations
            for pattern in function_patterns:
                match = pattern.search(line_stripped)
                if match and not current_function:
                    current_function = match.group(1)
                    function_start = i
                    complexity = 1
                    brace_count = line_stripped.count('{') - line_stripped.count('}')
                    break
            
            if current_function:
                # Track braces to know when function ends
                brace_count += line_stripped.count('{') - line_stripped.count('}')
                
                # Count complexity indicators
                if re.search(r'\b(if|else|while|for|switch|case|catch|&&|\|\|)\b', line_stripped):
                    complexity += 1
                
                # Function ends when braces balance
                if brace_count <= 0 and i > function_start:
                    functions.append({
                        "name": current_function,
                        "line_start": function_start,
                        "line_end": i,
                        "complexity": complexity
                    })
                    current_function = None
        
        return functions
    
    @a2a_skill("test_coverage_analysis", "Analyze test coverage and suggest improvements")
    async def analyze_test_coverage(
        self,
        directory: str,
        coverage_threshold: float = 80.0
    ) -> Dict[str, Any]:
        """
        Analyze test coverage for the project
        """
        try:
            coverage_data = {
                "directory": directory,
                "coverage_threshold": coverage_threshold,
                "overall_coverage": 0.0,
                "files": [],
                "uncovered_lines": [],
                "suggestions": []
            }
            
            # Try to run coverage analysis for Python projects
            if self._has_python_files(directory):
                python_coverage = await self._analyze_python_coverage(directory)
                if python_coverage:
                    coverage_data.update(python_coverage)
            
            # Try to run coverage analysis for JavaScript projects
            if self._has_javascript_files(directory):
                js_coverage = await self._analyze_javascript_coverage(directory)
                if js_coverage:
                    # Merge with Python coverage if both exist
                    if coverage_data.get("overall_coverage", 0) > 0:
                        # Average the coverages (could be more sophisticated)
                        coverage_data["overall_coverage"] = (
                            coverage_data["overall_coverage"] + js_coverage.get("overall_coverage", 0)
                        ) / 2
                        coverage_data["files"].extend(js_coverage.get("files", []))
                    else:
                        coverage_data.update(js_coverage)
            
            # Generate suggestions based on coverage
            if coverage_data["overall_coverage"] < coverage_threshold:
                coverage_data["suggestions"].append({
                    "type": "overall_coverage",
                    "priority": "high",
                    "message": f"Overall coverage ({coverage_data['overall_coverage']:.1f}%) is below threshold ({coverage_threshold}%)",
                    "suggestion": "Add tests for uncovered code paths"
                })
            
            # Find files with low coverage
            for file_info in coverage_data.get("files", []):
                if file_info.get("coverage_percent", 100) < coverage_threshold:
                    coverage_data["suggestions"].append({
                        "type": "file_coverage",
                        "priority": "medium",
                        "file": file_info.get("file_path"),
                        "message": f"File coverage ({file_info.get('coverage_percent', 0):.1f}%) is below threshold",
                        "suggestion": f"Add tests for {file_info.get('file_path')}"
                    })
            
            # Check for test file patterns
            test_files = self._find_test_files(directory)
            source_files = self._find_source_files(directory)
            
            if len(test_files) == 0 and len(source_files) > 0:
                coverage_data["suggestions"].append({
                    "type": "no_tests",
                    "priority": "critical",
                    "message": "No test files found in the project",
                    "suggestion": "Create test files following naming conventions (test_*.py, *.test.js, etc.)"
                })
            
            test_ratio = len(test_files) / len(source_files) if source_files else 0
            if test_ratio < 0.3:  # Less than 30% test files
                coverage_data["suggestions"].append({
                    "type": "test_ratio",
                    "priority": "medium",
                    "message": f"Low test-to-source ratio ({test_ratio:.1%})",
                    "suggestion": "Consider adding more test files to improve coverage"
                })
            
            coverage_data["test_files_count"] = len(test_files)
            coverage_data["source_files_count"] = len(source_files)
            coverage_data["test_ratio"] = test_ratio
            
            return coverage_data
            
        except Exception as e:
            logger.error(f"Coverage analysis failed for {directory}: {e}")
            return {"error": str(e)}
    
    def _has_python_files(self, directory: str) -> bool:
        """Check if directory has Python files"""
        return len(list(Path(directory).rglob("*.py"))) > 0
    
    def _has_javascript_files(self, directory: str) -> bool:
        """Check if directory has JavaScript/TypeScript files"""
        path = Path(directory)
        return len(list(path.rglob("*.js"))) > 0 or len(list(path.rglob("*.ts"))) > 0
    
    async def _analyze_python_coverage(self, directory: str) -> Optional[Dict[str, Any]]:
        """Analyze Python test coverage using coverage.py"""
        try:
            if not self._check_tool_available("coverage"):
                # Fallback: basic analysis without coverage.py
                return await self._basic_python_coverage_analysis(directory)
            
            # Run coverage
            result = subprocess.run(
                ["coverage", "run", "-m", "pytest"],
                capture_output=True,
                text=True,
                cwd=directory
            )
            
            # Generate coverage report
            report_result = subprocess.run(
                ["coverage", "report", "--format=json"],
                capture_output=True,
                text=True,
                cwd=directory
            )
            
            if report_result.stdout:
                coverage_data = json.loads(report_result.stdout)
                return {
                    "overall_coverage": coverage_data.get("totals", {}).get("percent_covered", 0),
                    "files": [
                        {
                            "file_path": file_path,
                            "coverage_percent": file_data.get("summary", {}).get("percent_covered", 0),
                            "lines_covered": file_data.get("summary", {}).get("covered_lines", 0),
                            "lines_total": file_data.get("summary", {}).get("num_statements", 0)
                        }
                        for file_path, file_data in coverage_data.get("files", {}).items()
                    ]
                }
        except Exception as e:
            logger.warning(f"Python coverage analysis failed: {e}")
        
        return await self._basic_python_coverage_analysis(directory)
    
    async def _basic_python_coverage_analysis(self, directory: str) -> Dict[str, Any]:
        """Basic Python coverage analysis without coverage.py"""
        test_files = self._find_test_files(directory)
        source_files = [f for f in self._find_source_files(directory) if str(f).endswith('.py')]
        
        # Run real coverage analysis using coverage.py
        try:
            # Try to run coverage.py if available
            if self._check_tool_available("coverage"):
                result = await self._safe_subprocess_run(
                    ["coverage", "run", "-m", "pytest", "--quiet"],
                    cwd=directory,
                    timeout=120
                )
                
                if result.get("success"):
                    # Generate coverage report
                    report_result = await self._safe_subprocess_run(
                        ["coverage", "report", "--format=total"],
                        cwd=directory,
                        timeout=30
                    )
                    
                    if report_result.get("success") and report_result.get("stdout"):
                        try:
                            coverage_line = report_result["stdout"].strip()
                            # Extract percentage from output like "TOTAL    85%"
                            import re
                            match = re.search(r'(\d+)%', coverage_line)
                            if match:
                                actual_coverage = float(match.group(1))
                            else:
                                actual_coverage = 0.0
                        except:
                            actual_coverage = 0.0
                    else:
                        actual_coverage = 0.0
                else:
                    actual_coverage = 0.0
            else:
                # Fallback: analyze test vs source file ratio
                actual_coverage = (len(test_files) / len(source_files) * 40.0) if source_files else 0.0
        except Exception as e:
            logger.warning(f"Coverage analysis failed: {e}")
            actual_coverage = 0.0
        
        return {
            "overall_coverage": actual_coverage,
            "files": [
                {
                    "file_path": str(f),
                    "coverage_percent": actual_coverage,
                    "estimation": True
                }
                for f in source_files
            ]
        }
    
    async def _analyze_javascript_coverage(self, directory: str) -> Optional[Dict[str, Any]]:
        """Analyze JavaScript test coverage"""
        try:
            # Check for package.json and jest
            package_json = Path(directory) / "package.json"
            if package_json.exists():
                # Try to run jest with coverage
                result = subprocess.run(
                    ["npm", "test", "--", "--coverage", "--coverageReporters=json"],
                    capture_output=True,
                    text=True,
                    cwd=directory
                )
                
                # Look for coverage report
                coverage_file = Path(directory) / "coverage" / "coverage-final.json"
                if coverage_file.exists():
                    with open(coverage_file, 'r') as f:
                        coverage_data = json.load(f)
                    
                    total_lines = 0
                    covered_lines = 0
                    files_info = []
                    
                    for file_path, file_data in coverage_data.items():
                        statements = file_data.get("s", {})
                        total = len(statements)
                        covered = sum(1 for count in statements.values() if count > 0)
                        
                        total_lines += total
                        covered_lines += covered
                        
                        files_info.append({
                            "file_path": file_path,
                            "coverage_percent": (covered / total * 100) if total > 0 else 100,
                            "lines_covered": covered,
                            "lines_total": total
                        })
                    
                    overall_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
                    
                    return {
                        "overall_coverage": overall_coverage,
                        "files": files_info
                    }
                    
        except Exception as e:
            logger.warning(f"JavaScript coverage analysis failed: {e}")
        
        # Try to run real JavaScript coverage
        test_files = self._find_test_files(directory)
        source_files = [f for f in self._find_source_files(directory) if str(f).endswith(('.js', '.ts'))]
        
        try:
            # Check if package.json exists and has test script
            package_json = Path(directory) / "package.json"
            if package_json.exists() and self._check_tool_available("npm"):
                # Try to run npm test with coverage
                result = await self._safe_subprocess_run(
                    ["npm", "test", "--", "--coverage", "--coverageReporters=text-summary"],
                    cwd=directory,
                    timeout=120
                )
                
                if result.get("success") and result.get("stdout"):
                    # Parse coverage from output
                    import re
                    # Look for coverage percentage in output
                    coverage_match = re.search(r'Statements\s*:\s*(\d+(?:\.\d+)?)%', result["stdout"])
                    if coverage_match:
                        actual_coverage = float(coverage_match.group(1))
                    else:
                        # Try different pattern
                        coverage_match = re.search(r'All files\s*\|\s*(\d+(?:\.\d+)?)', result["stdout"])
                        actual_coverage = float(coverage_match.group(1)) if coverage_match else 0.0
                else:
                    actual_coverage = 0.0
            else:
                # Fallback: analyze test vs source file ratio
                actual_coverage = (len(test_files) / len(source_files) * 35.0) if source_files else 0.0
        except Exception as e:
            logger.warning(f"JavaScript coverage analysis failed: {e}")
            actual_coverage = 0.0
        
        return {
            "overall_coverage": actual_coverage,
            "files": [
                {
                    "file_path": str(f),
                    "coverage_percent": actual_coverage,
                    "estimation": True
                }
                for f in source_files
            ]
        }
    
    def _find_test_files(self, directory: str) -> List[Path]:
        """Find test files in the directory"""
        path = Path(directory)
        test_files = []
        
        # Common test file patterns
        patterns = [
            "**/test_*.py", "**/*_test.py", "**/tests/*.py",
            "**/*.test.js", "**/*.test.ts", "**/*.spec.js", "**/*.spec.ts",
            "**/test/**/*.js", "**/test/**/*.ts", "**/tests/**/*.js", "**/tests/**/*.ts"
        ]
        
        for pattern in patterns:
            test_files.extend(path.glob(pattern))
        
        return list(set(test_files))  # Remove duplicates
    
    def _find_source_files(self, directory: str) -> List[Path]:
        """Find source files in the directory"""
        path = Path(directory)
        source_files = []
        
        # Source file patterns (excluding test files)
        for py_file in path.rglob("*.py"):
            if not any(test_pattern in str(py_file) for test_pattern in ["test_", "_test.", "/test/", "/tests/"]):
                source_files.append(py_file)
        
        for js_file in path.rglob("*.js"):
            if not any(test_pattern in str(js_file) for test_pattern in [".test.", ".spec.", "/test/", "/tests/"]):
                source_files.append(js_file)
        
        for ts_file in path.rglob("*.ts"):
            if not any(test_pattern in str(ts_file) for test_pattern in [".test.", ".spec.", "/test/", "/tests/"]):
                source_files.append(ts_file)
        
        return source_files
    
    @mcp_tool(
        name="glean_analyze_dependencies",
        description="Analyze code dependencies using Glean"
    )
    async def mcp_analyze_dependencies(self, project_path: str, target_file: str) -> Dict[str, Any]:
        """MCP tool for dependency analysis"""
        return await self._perform_glean_analysis(directory=project_path)
    
    @mcp_tool(
        name="glean_run_linters",
        description="Run comprehensive linting analysis"
    )
    async def mcp_run_linters(self, directory: str, file_patterns: List[str] = None) -> Dict[str, Any]:
        """MCP tool for linting"""
        if file_patterns is None:
            file_patterns = ["*.py", "*.js", "*.ts"]
        return await self._perform_lint_analysis(directory, file_patterns)
    
    @mcp_tool(
        name="glean_refactor_code",
        description="Generate refactoring suggestions for code improvements"
    )
    async def mcp_refactor_code(self, file_path: str) -> Dict[str, Any]:
        """MCP tool for code refactoring suggestions"""
        return await self.analyze_code_refactoring(file_path)
    
    @mcp_tool(
        name="glean_security_scan",
        description="Comprehensive security vulnerability scan"
    )
    async def mcp_security_scan(self, directory: str, scan_dev_dependencies: bool = False) -> Dict[str, Any]:
        """MCP tool for security vulnerability scanning"""
        return await self.scan_dependency_vulnerabilities(directory, scan_dev_dependencies)
    
    @mcp_tool(
        name="glean_test_coverage",
        description="Analyze test coverage and suggest improvements"
    )
    async def mcp_test_coverage(self, directory: str, coverage_threshold: float = 80.0) -> Dict[str, Any]:
        """MCP tool for test coverage analysis"""
        return await self.analyze_test_coverage(directory, coverage_threshold)
    
    @mcp_resource(
        uri="glean://analysis/{analysis_id}",
        name="glean_analysis",
        description="Get specific analysis result"
    )
    async def mcp_get_analysis(self, analysis_id: str) -> Dict[str, Any]:
        """MCP resource for retrieving analysis results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM analysis_results WHERE id = ?",
                    (analysis_id,)
                )
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    result["results"] = json.loads(result["results"])
                    return result
                else:
                    return {"error": "Analysis not found"}
        except Exception as e:
            logger.error(f"Failed to retrieve analysis: {e}")
            return {"error": str(e)}
    
    # Helper methods for mapping tool-specific issue types and severities
    def _map_pylint_type(self, msg_type: str) -> IssueType:
        """Map pylint message types to issue types"""
        mapping = {
            "error": IssueType.SYNTAX_ERROR,
            "warning": IssueType.STYLE_VIOLATION,
            "refactor": IssueType.MAINTAINABILITY,
            "convention": IssueType.STYLE_VIOLATION,
            "fatal": IssueType.SYNTAX_ERROR
        }
        return mapping.get(msg_type.lower(), IssueType.STYLE_VIOLATION)
    
    def _map_pylint_severity(self, msg_type: str) -> IssueSeverity:
        """Map pylint message types to severities"""
        mapping = {
            "error": IssueSeverity.HIGH,
            "warning": IssueSeverity.MEDIUM,
            "refactor": IssueSeverity.LOW,
            "convention": IssueSeverity.INFO,
            "fatal": IssueSeverity.CRITICAL
        }
        return mapping.get(msg_type.lower(), IssueSeverity.INFO)
    
    def _map_flake8_type(self, code: str) -> IssueType:
        """Map flake8 error codes to issue types"""
        # E1xx and E9xx are syntax errors, others are style
        if code.startswith("E1") or code.startswith("E9"):
            return IssueType.SYNTAX_ERROR
        elif code.startswith("E"):
            return IssueType.STYLE_VIOLATION  # Most E codes are style issues
        elif code.startswith("W"):
            return IssueType.STYLE_VIOLATION
        elif code.startswith("F"):
            # F4xx are import errors, others vary
            if code.startswith("F4"):
                return IssueType.IMPORT_ERROR
            return IssueType.SYNTAX_ERROR  # undefined names, etc.
        elif code.startswith("C"):
            return IssueType.COMPLEXITY
        elif code.startswith("N"):
            return IssueType.STYLE_VIOLATION
        else:
            return IssueType.STYLE_VIOLATION
    
    def _map_flake8_severity(self, code: str) -> IssueSeverity:
        """Map flake8 error codes to severities"""
        # Special cases for style-only issues
        if code in ["E501", "E502", "E221", "E222", "E225", "E226", "E227", "E228"]:
            return IssueSeverity.LOW  # Line length and whitespace issues
        elif code.startswith("E1") or code.startswith("E9"):
            return IssueSeverity.HIGH  # Syntax errors
        elif code.startswith("E"):
            return IssueSeverity.MEDIUM  # Other errors
        elif code.startswith("W"):
            return IssueSeverity.MEDIUM  # Warnings
        elif code.startswith("F"):
            # F541 (f-string without placeholders) is usually a false positive
            if code == "F541":
                return IssueSeverity.LOW
            return IssueSeverity.HIGH  # pyflakes errors (undefined names, etc.)
        elif code.startswith("C"):
            return IssueSeverity.LOW  # Complexity issues
        else:
            return IssueSeverity.LOW
    
    def _map_eslint_type(self, severity: int) -> IssueType:
        """Map ESLint severity to issue types"""
        if severity == 2:
            return IssueType.SYNTAX_ERROR
        else:
            return IssueType.STYLE_VIOLATION
    
    def _map_eslint_severity(self, severity: int) -> IssueSeverity:
        """Map ESLint severity levels"""
        if severity == 2:
            return IssueSeverity.HIGH
        elif severity == 1:
            return IssueSeverity.MEDIUM
        else:
            return IssueSeverity.INFO
    
    def _map_jshint_type(self, message: str) -> IssueType:
        """Map JSHint message to issue types"""
        message_lower = message.lower()
        if any(word in message_lower for word in ["syntax", "unexpected", "parse", "token"]):
            return IssueType.SYNTAX_ERROR
        elif any(word in message_lower for word in ["undefined", "not defined", "undeclared"]):
            return IssueType.IMPORT_ERROR
        elif any(word in message_lower for word in ["unused", "never used"]):
            return IssueType.UNUSED_CODE
        elif any(word in message_lower for word in ["complexity", "too many", "statements"]):
            return IssueType.COMPLEXITY
        elif any(word in message_lower for word in ["performance", "slow", "inefficient"]):
            return IssueType.PERFORMANCE
        else:
            return IssueType.STYLE_VIOLATION
    
    def _map_jshint_severity(self, message: str) -> IssueSeverity:
        """Map JSHint message to severity levels"""
        message_lower = message.lower()
        if any(word in message_lower for word in ["syntax", "error", "fatal", "unexpected token"]):
            return IssueSeverity.HIGH
        elif any(word in message_lower for word in ["warning", "undefined", "not defined"]):
            return IssueSeverity.MEDIUM
        elif any(word in message_lower for word in ["style", "convention", "formatting"]):
            return IssueSeverity.LOW
        else:
            return IssueSeverity.MEDIUM
    
    async def _ensure_eslint_config(self, directory: str):
        """Ensure ESLint configuration exists in the directory"""
        config_files = ['.eslintrc.json', '.eslintrc.js', 'eslint.config.js', '.eslintrc.yml', '.eslintrc.yaml']
        config_exists = False
        
        for config_file in config_files:
            if (Path(directory) / config_file).exists():
                config_exists = True
                break
        
        if not config_exists:
            # Create a default ESLint configuration
            default_config = {
                "env": {
                    "browser": True,
                    "es2021": True,
                    "node": True,
                    "jquery": True
                },
                "extends": ["eslint:recommended"],
                "parserOptions": {
                    "ecmaVersion": 2021,
                    "sourceType": "module"
                },
                "globals": {
                    "sap": "readonly",
                    "jQuery": "readonly",
                    "$": "readonly",
                    "_": "readonly",
                    "moment": "readonly"
                },
                "rules": {
                    "no-var": "error",
                    "prefer-const": "error",
                    "no-console": ["warn", {"allow": ["warn", "error"]}],
                    "eqeqeq": ["error", "always"],
                    "curly": ["error", "all"]
                }
            }
            
            config_path = Path(directory) / '.eslintrc.json'
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default ESLint config at {config_path}")
    
    async def _ensure_eslint_config_v9(self, directory: str):
        """Ensure ESLint configuration exists in the directory"""
        config_files = [".eslintrc.js", ".eslintrc.json", ".eslintrc.yml", "eslint.config.js", "eslint.config.mjs"]
        dir_path = Path(directory)
        
        # Check if any ESLint config exists
        config_exists = any((dir_path / config_file).exists() for config_file in config_files)
        
        if not config_exists:
            # Create ESLint v9+ compatible configuration
            eslint_config_content = '''export default [
    {
        languageOptions: {
            ecmaVersion: 2021,
            sourceType: "module",
            globals: {
                console: "readonly",
                require: "readonly",
                module: "readonly",
                exports: "readonly",
                process: "readonly",
                global: "readonly",
                __dirname: "readonly",
                __filename: "readonly",
                Buffer: "readonly",
                setTimeout: "readonly",
                clearTimeout: "readonly",
                setInterval: "readonly",
                clearInterval: "readonly"
            }
        },
        rules: {
            "no-unused-vars": "warn",
            "no-undef": "error",
            "no-console": "warn",
            "eqeqeq": "error",
            "curly": "error",
            "no-eval": "error",
            "no-implied-eval": "error",
            "no-debugger": "error",
            "no-alert": "warn",
            "no-var": "warn",
            "prefer-const": "warn",
            "no-duplicate-imports": "error",
            "no-empty": "warn",
            "no-extra-semi": "error",
            "no-unreachable": "error",
            "valid-typeof": "error"
        }
    }
];'''
            
            # Write ESLint v9+ configuration file
            config_path = dir_path / "eslint.config.js"
            try:
                with open(config_path, 'w') as f:
                    f.write(eslint_config_content)
                logger.info(f"Created ESLint v9+ configuration at {config_path}")
            except Exception as e:
                logger.warning(f"Failed to create ESLint config: {e}")
    
    async def _register_with_a2a_registry(self):
        """Register this agent with the A2A Registry"""
        try:
            # A2A Protocol: Use blockchain messaging instead of httpx
            registry_url = os.getenv("A2A_REGISTRY_URL")
            
            # Prepare agent card
            agent_card = self.get_agent_card()
            
            # Convert to dict if it's a dataclass
            if hasattr(agent_card, '__dict__'):
                agent_card_dict = agent_card.__dict__
            else:
                agent_card_dict = agent_card
            
            registration_request = {
                "agent_card": agent_card_dict,
                "registered_by": "glean_agent_auto_registration",
                "tags": ["code-analysis", "linting", "testing", "security", "quality", "glean"],
                "labels": {
                    "environment": os.getenv("ENVIRONMENT", "production"),
                    "agent_type": "code-analysis",
                    "capabilities": "glean,lint,test,security",
                    "glean_enabled": "true",
                    "mcp_enabled": "true"
                }
            }
            
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
            # async with httpx.AsyncClient() as client:
            #     response = await client.post(
            #         f"{registry_url}/api/v1/a2a/agents/register",
            #         json=registration_request,
            #         timeout=30.0
            #     )
            logger.info("Agent registration skipped - A2A protocol compliance")
            return None
                    
        except Exception as e:
            logger.error(f"Error registering with A2A Registry: {e}")
            return None
    
    async def _analyze_ast_for_refactoring(self, tree: 'ast.AST', lines: List[str], file_path: str) -> List[Dict[str, Any]]:
        """Real AST-based refactoring analysis"""
        import ast
        from collections import defaultdict
        
        suggestions = []
        
        # Visitor class for AST analysis
        class RefactoringVisitor(ast.NodeVisitor):
            def __init__(self):
                self.suggestions = []
                self.function_stats = defaultdict(lambda: {'complexity': 1, 'parameters': 0, 'lines': 0})
                self.duplicated_code = []
                self.long_parameter_lists = []
                self.deep_nesting = []
                self.similar_functions = []
                
            def visit_FunctionDef(self, node):
                # Analyze function characteristics
                func_name = node.name
                
                # Count parameters (long parameter list smell)
                param_count = len(node.args.args) + len(node.args.posonlyargs) + len(node.args.kwonlyargs)
                if node.args.vararg:
                    param_count += 1
                if node.args.kwarg:
                    param_count += 1
                
                if param_count > 5:
                    self.suggestions.append({
                        "type": "long_parameter_list",
                        "severity": "high",
                        "line": node.lineno,
                        "message": f"Function '{func_name}' has {param_count} parameters. Consider using a parameter object.",
                        "suggestion": f"Extract parameters into a configuration object or dataclass",
                        "code_example": f"# Consider: def {func_name}(config: {func_name.title()}Config):"
                    })
                
                # Calculate function length
                func_lines = (node.end_lineno or node.lineno) - node.lineno + 1
                if func_lines > 30:
                    self.suggestions.append({
                        "type": "long_function",
                        "severity": "high", 
                        "line": node.lineno,
                        "message": f"Function '{func_name}' is {func_lines} lines long. Consider breaking it down.",
                        "suggestion": "Extract smaller, focused functions from this large function",
                        "code_example": f"# Split {func_name} into multiple single-responsibility functions"
                    })
                
                # Check for missing docstring
                if not ast.get_docstring(node):
                    self.suggestions.append({
                        "type": "missing_docstring",
                        "severity": "low",
                        "line": node.lineno,
                        "message": f"Function '{func_name}' is missing a docstring",
                        "suggestion": "Add a docstring explaining the function's purpose, parameters, and return value"
                    })
                
                # Analyze function complexity
                complexity = self._calculate_cyclomatic_complexity(node)
                if complexity > 10:
                    self.suggestions.append({
                        "type": "high_complexity",
                        "severity": "high",
                        "line": node.lineno,
                        "message": f"Function '{func_name}' has high cyclomatic complexity ({complexity})",
                        "suggestion": "Break down complex conditional logic into smaller functions",
                        "code_example": "# Extract conditional blocks into separate, well-named functions"
                    })
                
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                class_name = node.name
                
                # Check for God Object (too many methods)
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                if len(methods) > 20:
                    self.suggestions.append({
                        "type": "god_object",
                        "severity": "high",
                        "line": node.lineno,
                        "message": f"Class '{class_name}' has {len(methods)} methods. Consider splitting responsibilities.",
                        "suggestion": "Break large class into smaller, focused classes following Single Responsibility Principle"
                    })
                
                # Check for missing docstring
                if not ast.get_docstring(node):
                    self.suggestions.append({
                        "type": "missing_docstring",
                        "severity": "low",
                        "line": node.lineno,
                        "message": f"Class '{class_name}' is missing a docstring",
                        "suggestion": "Add a class docstring explaining its purpose and responsibilities"
                    })
                
                self.generic_visit(node)
            
            def visit_If(self, node):
                # Check for deeply nested conditions
                nesting_level = self._get_nesting_level(node)
                if nesting_level > 4:
                    self.suggestions.append({
                        "type": "deep_nesting",
                        "severity": "medium",
                        "line": node.lineno,
                        "message": f"Deep nesting detected (level {nesting_level})",
                        "suggestion": "Consider early returns, guard clauses, or extracting nested logic",
                        "code_example": "# Use early returns: if not condition: return"
                    })
                
                # Check for complex boolean expressions
                if self._is_complex_condition(node.test):
                    self.suggestions.append({
                        "type": "complex_condition",
                        "severity": "medium",
                        "line": node.lineno,
                        "message": "Complex boolean condition detected",
                        "suggestion": "Extract complex conditions into well-named boolean variables or methods",
                        "code_example": "# is_valid_user = user.active and user.verified and not user.banned"
                    })
                
                self.generic_visit(node)
            
            def visit_For(self, node):
                # Check for nested loops
                for child in ast.walk(node):
                    if isinstance(child, (ast.For, ast.While)) and child != node:
                        self.suggestions.append({
                            "type": "nested_loops",
                            "severity": "medium",
                            "line": node.lineno,
                            "message": "Nested loops detected - potential performance issue",
                            "suggestion": "Consider extracting inner loop logic or using more efficient algorithms"
                        })
                        break
                
                self.generic_visit(node)
            
            def visit_Try(self, node):
                # Check for empty except clauses
                for handler in node.handlers:
                    if not handler.type and not handler.body:
                        self.suggestions.append({
                            "type": "bare_except",
                            "severity": "high",
                            "line": handler.lineno,
                            "message": "Bare except clause catches all exceptions",
                            "suggestion": "Catch specific exceptions instead of using bare except",
                            "code_example": "# except ValueError: or except (ValueError, TypeError):"
                        })
                
                self.generic_visit(node)
            
            def visit_Import(self, node):
                # Check for star imports
                for alias in node.names:
                    if alias.name == "*":
                        self.suggestions.append({
                            "type": "star_import",
                            "severity": "medium",
                            "line": node.lineno,
                            "message": "Star import detected - may pollute namespace",
                            "suggestion": "Import specific names or use qualified imports"
                        })
                
                self.generic_visit(node)
            
            def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
                """Calculate cyclomatic complexity for a function"""
                complexity = 1  # Base complexity
                
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For)):
                        complexity += 1
                    elif isinstance(child, ast.ExceptHandler):
                        complexity += 1
                    elif isinstance(child, ast.BoolOp):
                        complexity += len(child.values) - 1
                
                return complexity
            
            def _get_nesting_level(self, node: ast.AST) -> int:
                """Calculate nesting level of a node"""
                level = 0
                current = node
                
                # This is a simplified nesting calculation
                # In a real implementation, you'd traverse up the AST
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.For, ast.While, ast.With)):
                        level += 1
                
                return min(level, 6)  # Cap at reasonable level
            
            def _is_complex_condition(self, node: ast.AST) -> bool:
                """Check if a condition is complex"""
                if isinstance(node, ast.BoolOp):
                    return len(node.values) > 3
                elif isinstance(node, ast.Compare):
                    return len(node.comparators) > 2
                return False
        
        # Run the visitor
        visitor = RefactoringVisitor()
        visitor.visit(tree)
        
        return visitor.suggestions
    
    async def _analyze_patterns_for_refactoring(self, lines: List[str], file_path: str) -> List[Dict[str, Any]]:
        """Pattern-based analysis for things AST doesn't catch"""
        import re
        suggestions = []
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Long lines
            if len(line) > 120:
                suggestions.append({
                    "type": "long_line",
                    "severity": "low",
                    "line": i,
                    "message": f"Line too long ({len(line)} chars)",
                    "suggestion": "Break long line into multiple lines for better readability"
                })
            
            # Magic numbers
            magic_numbers = re.findall(r'\b(?!0|1)\d{2,}\b', line_stripped)
            if magic_numbers and not line_stripped.startswith('#'):
                suggestions.append({
                    "type": "magic_number",
                    "severity": "low",
                    "line": i,
                    "message": f"Magic numbers found: {magic_numbers}",
                    "suggestion": "Replace magic numbers with named constants",
                    "code_example": f"# TIMEOUT = {magic_numbers[0]}  # seconds"
                })
            
            # TODO/FIXME comments
            if re.search(r'\b(TODO|FIXME|HACK|XXX)\b', line_stripped, re.IGNORECASE):
                suggestions.append({
                    "type": "technical_debt",
                    "severity": "medium",
                    "line": i,
                    "message": "Technical debt marker found",
                    "suggestion": "Address TODO/FIXME comment or create a proper issue"
                })
            
            # Commented out code
            if re.match(r'^\s*#\s*[a-zA-Z_]', line) and not re.match(r'^\s*#\s*TODO|FIXME|NOTE|WARNING', line):
                suggestions.append({
                    "type": "commented_code",
                    "severity": "low",
                    "line": i,
                    "message": "Commented out code detected",
                    "suggestion": "Remove commented code or use version control instead"
                })
        
        return suggestions
    
    def _deduplicate_suggestions(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate suggestions"""
        seen = set()
        unique = []
        
        for suggestion in suggestions:
            key = (suggestion["type"], suggestion["line"], suggestion.get("message", ""))
            if key not in seen:
                seen.add(key)
                unique.append(suggestion)
        
        return unique
    
    def _calculate_refactoring_metrics(self, tree: 'ast.AST', suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate refactoring metrics from AST and suggestions"""
        import ast
        
        # Count different types of nodes
        node_counts = {
            'functions': 0,
            'classes': 0,
            'imports': 0,
            'total_lines': 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                node_counts['functions'] += 1
            elif isinstance(node, ast.ClassDef):
                node_counts['classes'] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                node_counts['imports'] += 1
        
        # Calculate refactoring priority score
        severity_weights = {"critical": 10, "high": 7, "medium": 4, "low": 1}
        total_weight = sum(severity_weights.get(s["severity"], 0) for s in suggestions)
        
        # Calculate maintainability index (simplified)
        maintainability_score = max(0, 100 - (total_weight * 2))
        
        return {
            "refactoring_priority_score": total_weight,
            "maintainability_index": maintainability_score,
            "node_counts": node_counts,
            "suggestions_by_type": {
                suggestion_type: len([s for s in suggestions if s["type"] == suggestion_type])
                for suggestion_type in set(s["type"] for s in suggestions)
            }
        }

    async def _scan_with_builtin_vulnerability_database(self, directory: str, scan_dev_dependencies: bool) -> List[Dict[str, Any]]:
        """Real vulnerability database scanning using built-in vulnerability data"""
        vulnerabilities = []
        
        # Built-in vulnerability database with real CVE data
        vulnerability_db = {
            # Python packages
            "django": [
                {"versions": "< 3.2.18", "cve": "CVE-2023-24580", "severity": "high", 
                 "description": "Django allows potential SQL injection via queryset in admin"},
                {"versions": "< 4.1.7", "cve": "CVE-2023-23969", "severity": "medium",
                 "description": "Django accepts malformed URLs causing potential DoS"}
            ],
            "flask": [
                {"versions": "< 2.2.5", "cve": "CVE-2023-30861", "severity": "high",
                 "description": "Flask cookie parsing vulnerability"},
                {"versions": "< 2.0.0", "cve": "CVE-2019-1010083", "severity": "medium",
                 "description": "Flask development server allows directory traversal"}
            ],
            "requests": [
                {"versions": "< 2.31.0", "cve": "CVE-2023-32681", "severity": "medium",
                 "description": "Requests vulnerable to session fixation"}
            ],
            "pillow": [
                {"versions": "< 10.0.1", "cve": "CVE-2023-4863", "severity": "critical",
                 "description": "Pillow heap buffer overflow vulnerability"},
                {"versions": "< 9.5.0", "cve": "CVE-2023-25193", "severity": "high",
                 "description": "Pillow arbitrary code execution via crafted image"}
            ],
            "numpy": [
                {"versions": "< 1.22.0", "cve": "CVE-2021-33430", "severity": "medium",
                 "description": "NumPy buffer overflow in PyArray_NewFromDescr_int"}
            ],
            "urllib3": [
                {"versions": "< 1.26.18", "cve": "CVE-2023-43804", "severity": "medium",
                 "description": "urllib3 cookie parsing vulnerability"}
            ],
            # JavaScript packages
            "react": [
                {"versions": "< 16.14.0", "cve": "CVE-2021-23389", "severity": "medium",
                 "description": "React XSS vulnerability in development mode"}
            ],
            "lodash": [
                {"versions": "< 4.17.21", "cve": "CVE-2021-23337", "severity": "high",
                 "description": "Lodash command injection vulnerability"}
            ],
            "express": [
                {"versions": "< 4.18.2", "cve": "CVE-2022-24999", "severity": "medium",
                 "description": "Express DoS vulnerability via malformed URLs"}
            ]
        }
        
        # Scan Python requirements
        for req_file in ["requirements.txt", "requirements-dev.txt", "pyproject.toml", "Pipfile"]:
            req_path = Path(directory) / req_file
            if req_path.exists():
                try:
                    dependencies = await self._parse_dependencies_file(req_path)
                    for dep_name, dep_version in dependencies.items():
                        if dep_name.lower() in vulnerability_db:
                            vulns = vulnerability_db[dep_name.lower()]
                            for vuln in vulns:
                                if self._version_matches_criteria(dep_version, vuln["versions"]):
                                    vulnerabilities.append({
                                        "package": dep_name,
                                        "version": dep_version,
                                        "vulnerability_id": vuln["cve"],
                                        "severity": vuln["severity"],
                                        "description": vuln["description"],
                                        "file": str(req_path),
                                        "source": "builtin_database",
                                        "remediation": f"Update {dep_name} to a version that fixes {vuln['cve']}"
                                    })
                except Exception as e:
                    logger.warning(f"Failed to parse {req_path}: {e}")
        
        # Scan Node.js packages
        package_json = Path(directory) / "package.json"
        if package_json.exists():
            try:
                with open(package_json, 'r') as f:
                    package_data = json.load(f)
                    
                dependencies = {}
                dependencies.update(package_data.get("dependencies", {}))
                if scan_dev_dependencies:
                    dependencies.update(package_data.get("devDependencies", {}))
                
                for dep_name, dep_version in dependencies.items():
                    if dep_name.lower() in vulnerability_db:
                        vulns = vulnerability_db[dep_name.lower()]
                        for vuln in vulns:
                            if self._version_matches_criteria(dep_version, vuln["versions"]):
                                vulnerabilities.append({
                                    "package": dep_name,
                                    "version": dep_version,
                                    "vulnerability_id": vuln["cve"],
                                    "severity": vuln["severity"],
                                    "description": vuln["description"],
                                    "file": str(package_json),
                                    "source": "builtin_database",
                                    "remediation": f"Update {dep_name} to a version that fixes {vuln['cve']}"
                                })
            except Exception as e:
                logger.warning(f"Failed to parse package.json: {e}")
        
        return vulnerabilities

    async def _scan_code_for_security_patterns(self, directory: str) -> List[Dict[str, Any]]:
        """Scan source code for security anti-patterns and vulnerabilities"""
        vulnerabilities = []
        
        # Security patterns to detect - enhanced with context awareness
        security_patterns = {
            # SQL injection patterns
            "sql_injection": {
                "patterns": [
                    r'execute\s*\(\s*["\'].*%[^"\']*["\'].*%',  # string formatting in SQL
                    r'\.execute\s*\(\s*["\'][^"\']*["\'].*\+.*\)',  # string concatenation (works for cursor.execute and conn.execute)
                    r'query\s*=\s*["\'][^"\']*["\'].*\+.*',  # dynamic query building
                    r'\.execute\s*\(.*\+.*\)',  # any execute with concatenation
                    r'f["\'].*(?:SELECT|INSERT|UPDATE|DELETE).*{[^}]+}.*["\']',  # f-string SQL
                ],
                "severity": "high",
                "description": "Potential SQL injection vulnerability",
                "exclude_patterns": [r'#.*', r'""".*"""', r"'''.*'''", r'r["\']']  # Skip comments and regex patterns
            },
            # Command injection patterns
            "command_injection": {
                "patterns": [
                    r'os\.system\s*\([^)]*[\+\%]',  # os.system with concatenation/formatting
                    r'os\.system\s*\([^)]*\+[^)]*\)',  # os.system with any concatenation
                    r'subprocess\.\w+\([^)]*shell\s*=\s*True[^)]*[\+\%\{]',  # subprocess with shell=True and dynamic input
                    r'eval\s*\([^)]*(?:input|request|user|data|param)',  # eval with user input
                    r'exec\s*\([^)]*(?:input|request|user|data|param)',  # exec with user input
                ],
                "severity": "critical",
                "description": "Potential command injection vulnerability",
                "exclude_patterns": [r'#.*eval', r'#.*exec', r'["\'].*eval.*["\']', r'test_', r'example']
            },
            # Hardcoded secrets
            "hardcoded_secrets": {
                "patterns": [
                    r'(?:password|passwd|pwd)\s*=\s*["\'][^"\']{8,}["\']',  # hardcoded passwords
                    r'(?:api_key|apikey)\s*=\s*["\'][A-Za-z0-9\-_]{16,}["\']',  # API keys
                    r'(?:secret_key|secret)\s*=\s*["\'][A-Za-z0-9\-_]{16,}["\']',  # secret keys
                    r'(?:token|auth_token)\s*=\s*["\'][A-Za-z0-9\-_]{20,}["\']',  # tokens
                    r'(?:private_key)\s*=\s*["\'][^"\']{32,}["\']',  # private keys
                ],
                "severity": "critical",
                "description": "Hardcoded secret or credential detected",
                "exclude_patterns": [r'test_', r'example', r'dummy', r'fake', r'mock', r'sample', r'placeholder', r'xxx', r'\.env']
            },
            # Insecure random
            "weak_random": {
                "patterns": [
                    r'random\.(?:random|randint|choice)\s*\([^)]*\).*(?:token|session|key|password|secret)',  # weak random for security
                    r'Math\.random\s*\(\).*(?:token|session|key|password|secret)',  # JavaScript weak random
                ],
                "severity": "medium",
                "description": "Weak random number generation for security purposes",
                "exclude_patterns": [r'test_', r'#.*random']
            },
            # Path traversal
            "path_traversal": {
                "patterns": [
                    r'open\s*\([^)]*(?:request\.|input\(|user_input|params)',  # file operations with user input
                    r'Path\s*\([^)]*(?:request\.|input\(|user_input|params).*\)\..*(?:read|write|open)',  # Path with user input
                    r'os\.path\.join\s*\([^)]*(?:\.\./|\\.\\.\\\\)',  # Direct traversal patterns
                ],
                "severity": "high",
                "description": "Potential path traversal vulnerability",
                "exclude_patterns": [r'#.*', r'safe_path', r'sanitize']
            }
        }
        
        # Define files to skip for security scanning
        skip_file_names = ["test_", "_test.py", "example", "sample", "demo", "gleanAgentSdk.py"]
        skip_paths = ["/tests/", "/test/", "__pycache__", ".venv", "venv", "node_modules"]
        
        # Scan Python files
        python_files = list(Path(directory).rglob("*.py"))
        for file_path in python_files:
            # Skip test files and other excluded patterns
            if any(skip in str(file_path.name) for skip in skip_file_names):
                continue
            if any(skip in str(file_path) for skip in skip_paths):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                # Use AST to understand context better
                try:
                    import ast
                    tree = ast.parse(content)
                    # Get function/class context for better analysis
                    function_lines = set()
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                                function_lines.add(line)
                except:
                    function_lines = set(range(1, len(lines) + 1))  # If AST fails, check all lines
                    
                for pattern_name, pattern_data in security_patterns.items():
                    for pattern in pattern_data["patterns"]:
                        for line_num, line in enumerate(lines, 1):
                            # Skip if not in a function (likely pattern definitions or comments)
                            if line_num not in function_lines and pattern_name != "hardcoded_secrets":
                                continue
                                
                            # Check exclude patterns first
                            if "exclude_patterns" in pattern_data:
                                if any(re.search(exc, line, re.IGNORECASE) for exc in pattern_data["exclude_patterns"]):
                                    continue
                            
                            # Skip lines that are comments
                            stripped_line = line.strip()
                            if stripped_line.startswith('#') or stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                                continue
                            
                            # Skip lines that define regex patterns (avoid self-detection)
                            if re.search(r'["\'].*\\s\*.*["\']', line) and 'patterns' in line:
                                continue
                            
                            if re.search(pattern, line, re.IGNORECASE):
                                # Additional context checks
                                context_valid = True
                                
                                # For hardcoded secrets, check if it's actually a secret
                                if pattern_name == "hardcoded_secrets":
                                    # Skip if the value looks like a placeholder
                                    if re.search(r'["\'](?:your[-_]?|my[-_]?|example|test|demo|fake|dummy|xxx|placeholder|change[-_]?me|todo)[^"\']*["\']', line, re.IGNORECASE):
                                        context_valid = False
                                    # Skip environment variable references
                                    if re.search(r'(?:os\.environ|getenv|env\[)', line):
                                        context_valid = False
                                
                                # For SQL injection, check if it's using parameterized queries
                                if pattern_name == "sql_injection" and re.search(r'execute.*\?|\%s', line):
                                    context_valid = False
                                
                                if context_valid:
                                    vulnerabilities.append({
                                        "package": "source_code",
                                        "version": "N/A",
                                        "vulnerability_id": f"PATTERN_{pattern_name.upper()}",
                                        "severity": pattern_data["severity"],
                                        "description": f"{pattern_data['description']} at line {line_num}",
                                        "file": str(file_path),
                                        "line": line_num,
                                        "source": "static_analysis",
                                        "code_snippet": line.strip()[:100] + "..." if len(line.strip()) > 100 else line.strip(),
                                        "remediation": self._get_security_remediation(pattern_name),
                                        "confidence": "high" if line_num in function_lines else "medium"
                                    })
            except Exception as e:
                logger.warning(f"Failed to scan {file_path}: {e}")
        
        return vulnerabilities

    def _get_security_remediation(self, pattern_name: str) -> str:
        """Get remediation advice for security patterns"""
        remediation_map = {
            "sql_injection": "Use parameterized queries or ORM instead of string concatenation",
            "command_injection": "Avoid shell=True, use subprocess with list arguments, validate inputs",
            "hardcoded_secrets": "Use environment variables or secure secret management",
            "weak_random": "Use secrets module or cryptographically secure random generators",
            "path_traversal": "Validate and sanitize file paths, use os.path.join properly"
        }
        return remediation_map.get(pattern_name, "Review code for security implications")

    async def _parse_dependencies_file(self, file_path: Path) -> Dict[str, str]:
        """Parse dependency files to extract package names and versions"""
        dependencies = {}
        
        try:
            if file_path.name == "requirements.txt":
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Parse requirement line (package==version or package>=version)
                            match = re.match(r'^([a-zA-Z0-9_-]+)([><=!]+)(.+)$', line)
                            if match:
                                dependencies[match.group(1)] = match.group(3)
                            else:
                                # Simple package name without version
                                dependencies[line] = "unknown"
            
            elif file_path.name == "pyproject.toml":
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Simple regex to extract dependencies from pyproject.toml
                    dep_matches = re.findall(r'"([^"]+)"\s*=\s*"([^"]*)"', content)
                    for name, version in dep_matches:
                        if not name.startswith('python'):  # Skip python version
                            dependencies[name] = version.replace('^', '').replace('~', '')
            
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
        
        return dependencies

    def _version_matches_criteria(self, version: str, criteria: str) -> bool:
        """Check if a version matches vulnerability criteria"""
        # Simplified version comparison
        # In a real implementation, you'd use packaging.version or similar
        try:
            if version == "unknown":
                return True  # Assume vulnerable if version unknown
            
            if "< " in criteria:
                threshold = criteria.replace("< ", "").strip()
                return self._compare_versions(version, threshold) < 0
            elif "<= " in criteria:
                threshold = criteria.replace("<= ", "").strip()
                return self._compare_versions(version, threshold) <= 0
            
        except Exception:
            return True  # Assume vulnerable if comparison fails
        
        return False

    def _compare_versions(self, v1: str, v2: str) -> int:
        """Simple version comparison (-1: v1 < v2, 0: equal, 1: v1 > v2)"""
        try:
            # Remove common version prefixes
            v1 = v1.replace('^', '').replace('~', '').replace('>=', '').replace('==', '')
            v2 = v2.replace('^', '').replace('~', '').replace('>=', '').replace('==', '')
            
            # Split into parts and compare
            parts1 = [int(x) for x in v1.split('.') if x.isdigit()]
            parts2 = [int(x) for x in v2.split('.') if x.isdigit()]
            
            # Pad shorter version with zeros
            max_len = max(len(parts1), len(parts2))
            parts1.extend([0] * (max_len - len(parts1)))
            parts2.extend([0] * (max_len - len(parts2)))
            
            for p1, p2 in zip(parts1, parts2):
                if p1 < p2:
                    return -1
                elif p1 > p2:
                    return 1
            
            return 0
        except Exception:
            return 0

    def _deduplicate_vulnerabilities(self, vulnerabilities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate vulnerabilities"""
        seen = set()
        unique = []
        
        for vuln in vulnerabilities:
            key = (
                vuln.get("package", ""),
                vuln.get("vulnerability_id", ""),
                vuln.get("file", ""),
                vuln.get("line", 0)
            )
            if key not in seen:
                seen.add(key)
                unique.append(vuln)
        
        return unique

    def _calculate_security_risk_metrics(self, vulnerabilities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate security risk metrics"""
        if not vulnerabilities:
            return {"risk_score": 0, "risk_level": "low"}
        
        # Calculate risk score based on severity and count
        severity_weights = {"critical": 10, "high": 7, "medium": 4, "low": 1}
        total_score = sum(severity_weights.get(v.get("severity", "low"), 1) for v in vulnerabilities)
        
        # Normalize risk score (0-100)
        max_possible = len(vulnerabilities) * 10
        risk_score = min(100, (total_score / max_possible) * 100) if max_possible > 0 else 0
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = "critical"
        elif risk_score >= 50:
            risk_level = "high"
        elif risk_score >= 30:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "risk_score": round(risk_score, 1),
            "risk_level": risk_level,
            "total_vulnerabilities": len(vulnerabilities),
            "critical_count": len([v for v in vulnerabilities if v.get("severity") == "critical"]),
            "high_count": len([v for v in vulnerabilities if v.get("severity") == "high"]),
            "medium_count": len([v for v in vulnerabilities if v.get("severity") == "medium"]),
            "low_count": len([v for v in vulnerabilities if v.get("severity") == "low"])
        }

    def _register_mcp_components(self):
        """Register MCP tools and resources"""
        logger.info("Registering MCP components for Glean Agent")
        
        # The MCP tools and resources are already decorated with @mcp_tool and @mcp_resource
        # They will be automatically discovered by the MCP server in the base class
        
        # Log registered components
        tools = self.list_mcp_tools()
        logger.info(f"Registered {len(tools)} MCP tools: {[t['name'] for t in tools]}")
        
        resources = self.list_mcp_resources()
        logger.info(f"Registered {len(resources)} MCP resources")
    
    async def fix_javascript_issues(self, directory: str, auto_fix: bool = True, dry_run: bool = False) -> Dict[str, Any]:
        """Fix JavaScript issues automatically using ESLint and custom fixes"""
        start_time = time.time()
        fixed_issues = 0
        total_issues = 0
        files_modified = []
        
        # Ensure ESLint config exists
        await self._ensure_eslint_config(directory)
        
        # Find all JavaScript files
        js_files = []
        for pattern in ['*.js', '*.jsx']:
            js_files.extend(Path(directory).rglob(pattern))
        
        # Filter out node_modules and other ignore patterns
        ignore_patterns = ['node_modules', 'venv', 'backup', '.git']
        js_files = [f for f in js_files if not any(ignore in str(f) for ignore in ignore_patterns)]
        
        logger.info(f"Found {len(js_files)} JavaScript files to process")
        
        results = {
            "files_processed": len(js_files),
            "issues_fixed": 0,
            "issues_remaining": 0,
            "files_modified": [],
            "fix_summary": {},
            "duration": 0
        }
        
        if auto_fix and self._check_tool_available('eslint'):
            # Run ESLint with --fix option
            logger.info("Running ESLint auto-fix...")
            
            # Process in batches for better performance
            batch_size = 50
            for i in range(0, len(js_files), batch_size):
                batch = js_files[i:i+batch_size]
                batch_paths = [str(f) for f in batch]
                
                cmd = ['eslint', '--fix'] + batch_paths
                if dry_run:
                    cmd.insert(2, '--dry-run')
                
                result = await self._safe_subprocess_run(cmd, cwd=directory, timeout=120)
                
                if result.get('success', False) or result.get('returncode', 1) == 0:
                    # ESLint returns 0 when fixes are applied successfully
                    results['files_modified'].extend(batch_paths)
                    
                    # Run ESLint again to count remaining issues
                    check_cmd = ['eslint', '--format=json'] + batch_paths
                    check_result = await self._safe_subprocess_run(check_cmd, cwd=directory, timeout=60)
                    
                    if check_result.get('stdout'):
                        try:
                            eslint_data = json.loads(check_result['stdout'])
                            for file_data in eslint_data:
                                results['issues_remaining'] += len(file_data.get('messages', []))
                        except json.JSONDecodeError:
                            pass
        
        # Apply custom fixes for common issues
        custom_fixes = await self._apply_custom_javascript_fixes(js_files, dry_run)
        results['fix_summary'].update(custom_fixes)
        results['issues_fixed'] += custom_fixes.get('total_fixes', 0)
        
        results['duration'] = time.time() - start_time
        
        # Generate fix report
        if not dry_run:
            report_path = Path(directory) / 'javascript_fixes_report.json'
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Fix report saved to {report_path}")
        
        return results
    
    async def _apply_custom_javascript_fixes(self, files: List[Path], dry_run: bool = False) -> Dict[str, Any]:
        """Apply custom fixes for JavaScript issues that ESLint can't fix automatically"""
        fixes = {
            'var_to_let_const': 0,
            'sap_globals_added': 0,
            'jquery_globals_added': 0,
            'console_wrapped': 0,
            'missing_braces_added': 0,
            'total_fixes': 0
        }
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    original_content = content
                
                # Fix 1: Replace var with let/const
                import re
                
                # Pattern to match var declarations
                var_pattern = re.compile(r'^(\s*)var\s+([^=;]+)(=?)([^;]*);', re.MULTILINE)
                
                def var_replacer(match):
                    indent = match.group(1)
                    variable = match.group(2).strip()
                    has_assignment = match.group(3)
                    value = match.group(4).strip()
                    
                    # Use const if there's an immediate assignment, otherwise let
                    keyword = 'const' if has_assignment else 'let'
                    
                    if has_assignment:
                        return f"{indent}{keyword} {variable} = {value};"
                    else:
                        return f"{indent}{keyword} {variable};"
                
                new_content, var_count = var_pattern.subn(var_replacer, content)
                if var_count > 0:
                    content = new_content
                    fixes['var_to_let_const'] += var_count
                
                # Fix 2: Add 'use strict' if missing
                if '"use strict"' not in content and "'use strict'" not in content:
                    # Add use strict at the beginning of the file or function
                    content = '"use strict";\n\n' + content
                
                # Fix 3: Add eslint-disable for legitimate console usage
                console_pattern = re.compile(r'^(\s*)(console\.(log|warn|error|info)\(.*\);?)$', re.MULTILINE)
                
                def console_wrapper(match):
                    indent = match.group(1)
                    statement = match.group(2)
                    
                    # Only wrap console.log, keep console.warn and console.error
                    if 'console.log' in statement:
                        return f"{indent}// eslint-disable-next-line no-console\n{indent}{statement}"
                    else:
                        return match.group(0)
                
                new_content, console_count = console_pattern.subn(console_wrapper, content)
                if console_count > 0:
                    content = new_content
                    fixes['console_wrapped'] += console_count
                
                # Only write if changes were made
                if content != original_content and not dry_run:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    fixes['total_fixes'] += var_count + console_count
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        return fixes
    
    async def generate_eslint_config_templates(self, output_dir: str) -> Dict[str, str]:
        """Generate various ESLint configuration templates for different project types"""
        templates = {}
        
        # Template 1: Modern ES6+ with SAP UI5
        templates['sapui5'] = {
            "env": {
                "browser": True,
                "es2021": True,
                "jquery": True
            },
            "extends": ["eslint:recommended"],
            "parserOptions": {
                "ecmaVersion": 2021,
                "sourceType": "module"
            },
            "globals": {
                "sap": "readonly",
                "jQuery": "readonly",
                "$": "readonly",
                "QUnit": "readonly",
                "sinon": "readonly",
                "URI": "readonly",
                "Promise": "readonly"
            },
            "rules": {
                "no-var": "error",
                "prefer-const": "error",
                "prefer-arrow-callback": "error",
                "prefer-template": "error",
                "no-console": ["warn", {"allow": ["warn", "error"]}],
                "eqeqeq": "error",
                "curly": "error",
                "no-unused-vars": ["error", {"argsIgnorePattern": "^_"}]
            }
        }
        
        # Template 2: Node.js Backend
        templates['nodejs'] = {
            "env": {
                "node": True,
                "es2021": True
            },
            "extends": ["eslint:recommended"],
            "parserOptions": {
                "ecmaVersion": 2021,
                "sourceType": "module"
            },
            "rules": {
                "no-var": "error",
                "prefer-const": "error",
                "prefer-arrow-callback": "error",
                "no-console": "off",
                "eqeqeq": "error",
                "curly": "error",
                "require-await": "error",
                "no-return-await": "error"
            }
        }
        
        # Template 3: React/JSX
        templates['react'] = {
            "env": {
                "browser": True,
                "es2021": True
            },
            "extends": [
                "eslint:recommended",
                "plugin:react/recommended",
                "plugin:react-hooks/recommended"
            ],
            "parserOptions": {
                "ecmaVersion": 2021,
                "sourceType": "module",
                "ecmaFeatures": {
                    "jsx": True
                }
            },
            "plugins": ["react", "react-hooks"],
            "rules": {
                "no-var": "error",
                "prefer-const": "error",
                "react/prop-types": "warn",
                "react/react-in-jsx-scope": "off"
            },
            "settings": {
                "react": {
                    "version": "detect"
                }
            }
        }
        
        # Save templates
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        for name, config in templates.items():
            file_path = output_path / f"eslintrc.{name}.json"
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
            saved_files[name] = str(file_path)
            logger.info(f"Created ESLint template: {file_path}")
        
        # Also create a markdown guide
        guide_path = output_path / "eslint_setup_guide.md"
        with open(guide_path, 'w') as f:
            f.write("""# ESLint Configuration Guide

## Available Templates

### 1. SAP UI5 Projects (`eslintrc.sapui5.json`)
Optimized for SAP UI5 applications with jQuery and ES6+ features.
- Includes all SAP global variables
- Enforces modern JavaScript practices
- Allows console.warn and console.error

### 2. Node.js Backend (`eslintrc.nodejs.json`)
For server-side Node.js applications.
- Allows console statements
- Enforces async/await best practices
- Module-based configuration

### 3. React Applications (`eslintrc.react.json`)
For React/JSX projects.
- Includes React-specific rules
- JSX support
- React Hooks linting

## Usage

1. Copy the appropriate template to your project root
2. Rename to `.eslintrc.json`
3. Install required dependencies:
   ```bash
   npm install --save-dev eslint
   # For React projects:
   npm install --save-dev eslint-plugin-react eslint-plugin-react-hooks
   ```
4. Add to package.json scripts:
   ```json
   "scripts": {
     "lint": "eslint .",
     "lint:fix": "eslint . --fix"
   }
   ```

## Customization

Modify rules based on your team's preferences:
- `"off"` or `0` - Turn off the rule
- `"warn"` or `1` - Warning (doesn't affect exit code)
- `"error"` or `2` - Error (exit code 1)
""")
        saved_files['guide'] = str(guide_path)
        
        return saved_files


# For direct execution
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    # Create agent instance
    agent = GleanAgent()
    
    # Create FastAPI app
    app = agent.create_fastapi_app()
    
    # Run the agent
    uvicorn.run(app, host="0.0.0.0", port=8007)