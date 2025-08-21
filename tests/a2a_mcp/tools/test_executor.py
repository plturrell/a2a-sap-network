"""
A2A Test Executor - MCP Tool Implementation
Advanced test execution with intelligent orchestration
"""

import asyncio
import json
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestResult:
    """Test execution result."""
    name: str
    status: TestStatus
    duration: float
    output: str
    error: Optional[str] = None
    coverage: Optional[Dict[str, float]] = None

@dataclass
class TestSuite:
    """Test suite definition."""
    name: str
    type: str
    module: str
    tests: List[str]
    dependencies: List[str] = None

class TestExecutor:
    """Advanced test executor with parallel execution and intelligent orchestration."""
    
    def __init__(self, test_root: Path):
        self.test_root = test_root
        self.results_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def execute_test_suite(
        self,
        suite: TestSuite,
        parallel: bool = True,
        timeout: int = 300,
        coverage: bool = False
    ) -> List[TestResult]:
        """Execute a complete test suite."""
        logger.info(f"Executing test suite: {suite.name}")
        
        results = []
        start_time = time.time()
        
        if parallel and len(suite.tests) > 1:
            # Execute tests in parallel
            tasks = []
            for test in suite.tests:
                task = asyncio.create_task(
                    self._execute_single_test(test, suite, timeout, coverage)
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append(TestResult(
                        name=suite.tests[i],
                        status=TestStatus.ERROR,
                        duration=0.0,
                        output="",
                        error=str(result)
                    ))
                else:
                    final_results.append(result)
            results = final_results
        else:
            # Execute tests sequentially
            for test in suite.tests:
                result = await self._execute_single_test(test, suite, timeout, coverage)
                results.append(result)
        
        total_duration = time.time() - start_time
        logger.info(f"Suite {suite.name} completed in {total_duration:.2f}s")
        
        return results
    
    async def _execute_single_test(
        self,
        test_name: str,
        suite: TestSuite,
        timeout: int,
        coverage: bool
    ) -> TestResult:
        """Execute a single test."""
        start_time = time.time()
        
        try:
            # Build command based on test type
            cmd = self._build_test_command(test_name, suite, coverage)
            cwd = self._get_working_directory(suite)
            
            # Execute test
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                
                return TestResult(
                    name=test_name,
                    status=TestStatus.ERROR,
                    duration=time.time() - start_time,
                    output="",
                    error=f"Test timed out after {timeout}s"
                )
            
            # Parse results
            duration = time.time() - start_time
            output = stdout.decode() if stdout else ""
            error = stderr.decode() if stderr else ""
            
            # Determine status based on output and return code
            status = self._parse_test_status(test_name, process.returncode, output, error)
            
            # Extract coverage if available
            coverage_data = None
            if coverage:
                coverage_data = self._extract_coverage_data(output)
            
            return TestResult(
                name=test_name,
                status=status,
                duration=duration,
                output=output,
                error=error if error else None,
                coverage=coverage_data
            )
            
        except Exception as e:
            return TestResult(
                name=test_name,
                status=TestStatus.ERROR,
                duration=time.time() - start_time,
                output="",
                error=str(e)
            )
    
    def _build_test_command(self, test_name: str, suite: TestSuite, coverage: bool) -> List[str]:
        """Build command for test execution."""
        cmd = []
        
        if suite.type == "unit" or suite.type == "integration":
            if test_name.endswith('.py'):
                # Python/pytest tests
                cmd = ["/opt/homebrew/bin/python3.11", "-m", "pytest"]
                
                # Find the correct test path
                test_path = self.test_root / suite.type / suite.module / test_name
                if not test_path.exists():
                    test_path = self.test_root / suite.type / test_name
                
                cmd.append(str(test_path))
                
                # Add JSON output for better parsing
                cmd.extend(["--tb=short", "-v", "--no-header"])
                
                # Add coverage
                if coverage:
                    cmd.extend(["--cov=.", "--cov-report=term-missing"])
                    
            elif test_name.endswith('.js'):
                # JavaScript/Jest tests  
                cmd = ["npx", "jest", test_name, "--verbose", "--no-coverage"]
                if coverage:
                    cmd = ["npx", "jest", test_name, "--verbose", "--coverage"]
            else:
                # Default fallback
                cmd = ["/opt/homebrew/bin/python3.11", "-m", "pytest", str(self.test_root / suite.type / test_name), "-v"]
            
        elif suite.type == "e2e":
            if test_name.endswith(".cy.js"):
                cmd = ["npx", "cypress", "run", "--spec", test_name, "--reporter=json"]
            else:
                cmd = ["npm", "run", "test:e2e", "--", test_name]
        
        elif suite.type == "contracts":
            cmd = ["forge", "test", "--match-test", test_name.replace(".t.sol", ""), "-vv"]
            if coverage:
                cmd.extend(["--gas-report"])
        
        elif suite.type == "performance":
            cmd = ["/opt/homebrew/bin/python3.11", "-m", "pytest", str(self.test_root / "performance" / test_name)]
            cmd.extend(["--benchmark-only", "-v", "--tb=short"])
        
        elif suite.type == "security":
            cmd = ["/opt/homebrew/bin/python3.11", "-m", "pytest", str(self.test_root / "security" / test_name)]
            cmd.extend(["-v", "--tb=short"])
        
        return cmd
    
    def _parse_test_status(self, test_name: str, return_code: int, output: str, error: str) -> TestStatus:
        """Parse test status from command output."""
        
        # Check for explicit test results in output
        output_lower = output.lower()
        error_lower = error.lower()
        
        # Pytest patterns
        if "passed" in output_lower and return_code == 0:
            return TestStatus.PASSED
        elif "failed" in output_lower or "error" in output_lower:
            return TestStatus.FAILED
        elif "skipped" in output_lower:
            return TestStatus.SKIPPED
            
        # Jest patterns  
        if "pass" in output_lower and return_code == 0:
            return TestStatus.PASSED
        elif "fail" in output_lower:
            return TestStatus.FAILED
            
        # Forge/Solidity patterns
        if "test result: ok" in output_lower:
            return TestStatus.PASSED
        elif "test result: failed" in output_lower:
            return TestStatus.FAILED
            
        # Generic patterns
        if return_code == 0 and not any(word in output_lower for word in ['fail', 'error', 'exception']):
            return TestStatus.PASSED
        elif return_code != 0:
            return TestStatus.FAILED
        else:
            return TestStatus.ERROR
    
    def _get_working_directory(self, suite: TestSuite) -> Path:
        """Get working directory for test execution."""
        if suite.type == "contracts":
            return self.test_root.parent / "a2aNetwork"
        elif suite.type == "e2e" and suite.module == "a2aNetwork":
            return self.test_root.parent / "a2aNetwork"
        else:
            return self.test_root
    
    def _extract_coverage_data(self, output: str) -> Optional[Dict[str, float]]:
        """Extract coverage data from test output."""
        try:
            # Look for pytest-cov coverage output
            lines = output.split('\n')
            for line in lines:
                if 'TOTAL' in line and '%' in line:
                    # Parse coverage percentage
                    parts = line.split()
                    for part in parts:
                        if part.endswith('%'):
                            coverage_pct = float(part[:-1])
                            return {"total": coverage_pct}
            
            # Look for JSON coverage report
            if ".coverage" in output:
                # Coverage data is in separate file
                coverage_file = self.test_root / ".coverage"
                if coverage_file.exists():
                    # Would need coverage.py to parse this properly
                    return {"total": 0.0}
            
        except Exception as e:
            logger.warning(f"Failed to extract coverage data: {e}")
        
        return None

class TestSuiteBuilder:
    """Build test suites from the test directory structure."""
    
    def __init__(self, test_root: Path):
        self.test_root = test_root
    
    def build_suites(self, test_type: str = "all", module: str = "all") -> List[TestSuite]:
        """Build test suites based on criteria."""
        suites = []
        
        if test_type == "all":
            test_types = ["unit", "integration", "e2e", "performance", "security", "contracts"]
        else:
            test_types = [test_type]
        
        for t_type in test_types:
            type_suites = self._build_suites_for_type(t_type, module)
            suites.extend(type_suites)
        
        return suites
    
    def _build_suites_for_type(self, test_type: str, module: str) -> List[TestSuite]:
        """Build suites for a specific test type."""
        suites = []
        test_dir = self.test_root / test_type
        
        if not test_dir.exists():
            return suites
        
        if module == "all":
            # Find all modules in test type
            modules = [d.name for d in test_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            if not modules:
                modules = [""]  # Use root level
        else:
            modules = [module]
        
        for mod in modules:
            mod_dir = test_dir / mod if mod else test_dir
            
            if not mod_dir.exists():
                continue
            
            # Find test files
            test_files = []
            
            # Python tests
            test_files.extend([f.name for f in mod_dir.rglob("test_*.py")])
            test_files.extend([f.name for f in mod_dir.rglob("*_test.py")])
            
            # JavaScript tests
            test_files.extend([f.name for f in mod_dir.rglob("*.test.js")])
            test_files.extend([f.name for f in mod_dir.rglob("*.cy.js")])
            
            # Solidity tests
            test_files.extend([f.name for f in mod_dir.rglob("*.t.sol")])
            
            if test_files:
                suite_name = f"{test_type}_{mod}" if mod else test_type
                suite = TestSuite(
                    name=suite_name,
                    type=test_type,
                    module=mod,
                    tests=sorted(test_files)
                )
                suites.append(suite)
        
        return suites

class TestReporter:
    """Generate comprehensive test reports."""
    
    def __init__(self, test_root: Path):
        self.test_root = test_root
    
    def generate_summary_report(self, results: List[TestResult]) -> str:
        """Generate a summary report of test results."""
        total_tests = len(results)
        passed = len([r for r in results if r.status == TestStatus.PASSED])
        failed = len([r for r in results if r.status == TestStatus.FAILED])
        errors = len([r for r in results if r.status == TestStatus.ERROR])
        skipped = len([r for r in results if r.status == TestStatus.SKIPPED])
        
        total_duration = sum(r.duration for r in results)
        
        report = f"""
TEST EXECUTION SUMMARY
{'=' * 50}

Total Tests: {total_tests}
Passed:      {passed} ({passed/total_tests*100:.1f}%)
Failed:      {failed} ({failed/total_tests*100:.1f}%)
Errors:      {errors} ({errors/total_tests*100:.1f}%)
Skipped:     {skipped} ({skipped/total_tests*100:.1f}%)

Total Duration: {total_duration:.2f}s
Average Duration: {total_duration/total_tests:.2f}s per test

"""
        
        # Add failed tests details
        if failed > 0 or errors > 0:
            report += "\nFAILED/ERROR TESTS:\n"
            report += "-" * 30 + "\n"
            
            for result in results:
                if result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                    report += f"\n{result.name}: {result.status.value}\n"
                    if result.error:
                        report += f"  Error: {result.error[:200]}...\n"
        
        # Add coverage summary if available
        coverage_results = [r for r in results if r.coverage]
        if coverage_results:
            avg_coverage = sum(r.coverage.get("total", 0) for r in coverage_results) / len(coverage_results)
            report += f"\nCOVERAGE SUMMARY:\n"
            report += f"Average Coverage: {avg_coverage:.1f}%\n"
        
        return report
    
    def generate_json_report(self, results: List[TestResult]) -> str:
        """Generate a JSON report of test results."""
        report_data = {
            "timestamp": time.time(),
            "summary": {
                "total": len(results),
                "passed": len([r for r in results if r.status == TestStatus.PASSED]),
                "failed": len([r for r in results if r.status == TestStatus.FAILED]),
                "errors": len([r for r in results if r.status == TestStatus.ERROR]),
                "skipped": len([r for r in results if r.status == TestStatus.SKIPPED]),
            },
            "results": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "duration": r.duration,
                    "error": r.error,
                    "coverage": r.coverage
                }
                for r in results
            ]
        }
        
        return json.dumps(report_data, indent=2)
    
    def save_report(self, results: List[TestResult], format_type: str = "json") -> Path:
        """Save test report to file."""
        results_dir = self.test_root.parent / "test-results"
        results_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        
        if format_type == "json":
            filename = f"test_report_{timestamp}.json"
            content = self.generate_json_report(results)
        else:
            filename = f"test_report_{timestamp}.txt"
            content = self.generate_summary_report(results)
        
        report_file = results_dir / filename
        
        with open(report_file, 'w') as f:
            f.write(content)
        
        return report_file