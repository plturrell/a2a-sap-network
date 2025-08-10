#!/usr/bin/env python3
"""
Run integration tests between a2aAgents and a2aNetwork
"""

import subprocess
import sys
import os
from pathlib import Path
import time

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_header(message):
    """Print colored header"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{message:^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")


def run_command(cmd, description):
    """Run command and return success status"""
    print(f"{YELLOW}➤ {description}{RESET}")
    print(f"  Command: {cmd}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            print(f"{GREEN}✅ Success{RESET}")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"{RED}❌ Failed{RESET}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return False
    except Exception as e:
        print(f"{RED}❌ Exception: {e}{RESET}")
        return False


def check_prerequisites():
    """Check if prerequisites are met"""
    print_header("Checking Prerequisites")
    
    checks = [
        ("a2aAgents project exists", Path("/Users/apple/projects/a2a/a2aAgents").exists()),
        ("a2aNetwork project exists", Path("/Users/apple/projects/a2a/a2aNetwork").exists()),
        ("pytest installed", run_command("pytest --version", "Check pytest")),
        ("Python 3.8+", sys.version_info >= (3, 8))
    ]
    
    all_passed = True
    for check_name, passed in checks:
        if passed:
            print(f"{GREEN}✅ {check_name}{RESET}")
        else:
            print(f"{RED}❌ {check_name}{RESET}")
            all_passed = False
    
    return all_passed


def run_unit_tests():
    """Run unit tests"""
    print_header("Running Unit Tests")
    
    return run_command(
        "pytest tests/unit/test_network_components.py -v",
        "Unit tests for network components"
    )


def run_integration_tests():
    """Run integration tests"""
    print_header("Running Integration Tests")
    
    return run_command(
        "pytest tests/integration/test_a2a_network_integration.py -v -s",
        "Integration tests between a2aAgents and a2aNetwork"
    )


def run_quick_validation():
    """Run quick validation tests"""
    print_header("Quick Validation Tests")
    
    # Test imports work
    test_script = '''
import sys
sys.path.insert(0, "/Users/apple/projects/a2a/a2aAgents/backend")
sys.path.insert(0, "/Users/apple/projects/a2a/a2aNetwork")

# Test SDK imports
try:
    from app.a2a.sdk import A2AAgentBase, A2AMessage
    print("✅ SDK imports successful")
except Exception as e:
    print(f"❌ SDK import failed: {e}")
    exit(1)

# Test security imports  
try:
    from app.a2a.security import sign_a2a_message
    print("✅ Security imports successful")
except Exception as e:
    print(f"❌ Security import failed: {e}")
    exit(1)

# Test agent imports
try:
    from app.a2a.agents.agent0DataProduct.active.dataProductAgentSdk import DataProductRegistrationAgentSDK
    print("✅ Agent imports successful")
except Exception as e:
    print(f"❌ Agent import failed: {e}")
    exit(1)

print("\n✅ All quick validation tests passed!")
'''
    
    return run_command(
        f'python3 -c "{test_script}"',
        "Quick import validation"
    )


def generate_test_report():
    """Generate test report"""
    print_header("Generating Test Report")
    
    # Run pytest with coverage and HTML report
    run_command(
        "pytest tests/ --cov=app.a2a.network --cov=app.a2a.version " +
        "--cov-report=html:tests/coverage_report --cov-report=term",
        "Generate coverage report"
    )
    
    report_path = Path("tests/coverage_report/index.html")
    if report_path.exists():
        print(f"{GREEN}✅ Coverage report generated at: {report_path}{RESET}")
    else:
        print(f"{YELLOW}⚠️  Coverage report not generated{RESET}")


def main():
    """Main test runner"""
    print_header("A2A Integration Test Suite")
    print(f"Testing integration between a2aAgents and a2aNetwork")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    if not check_prerequisites():
        print(f"\n{RED}Prerequisites check failed. Please fix issues above.{RESET}")
        return 1
    
    # Run tests
    results = {
        "Quick Validation": run_quick_validation(),
        "Unit Tests": run_unit_tests(),
        "Integration Tests": run_integration_tests()
    }
    
    # Generate report
    generate_test_report()
    
    # Summary
    print_header("Test Summary")
    
    total_tests = len(results)
    passed_tests = sum(1 for passed in results.values() if passed)
    
    for test_name, passed in results.items():
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"{status} {test_name}")
    
    print(f"\n{BLUE}Total: {passed_tests}/{total_tests} test suites passed{RESET}")
    
    if passed_tests == total_tests:
        print(f"\n{GREEN}🎉 All integration tests passed!{RESET}")
        print(f"{GREEN}✅ a2aAgents and a2aNetwork are properly integrated{RESET}")
        return 0
    else:
        print(f"\n{RED}❌ Some tests failed. Review output above.{RESET}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
