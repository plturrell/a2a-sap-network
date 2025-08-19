#!/usr/bin/env python3
"""
A2A Test MCP CLI Tool
Command-line interface for the A2A Test MCP toolset
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add the tests directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.tools.test_executor import TestExecutor, TestSuiteBuilder, TestReporter
from mcp.agents.test_orchestrator import TestOrchestrator, TestPriority

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class A2ATestCLI:
    """Command-line interface for A2A Test MCP toolset."""
    
    def __init__(self, test_root: Path):
        self.test_root = test_root
        self.executor = TestExecutor(test_root)
        self.suite_builder = TestSuiteBuilder(test_root)
        self.reporter = TestReporter(test_root)
        self.orchestrator = TestOrchestrator(test_root)
    
    async def run_tests(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Execute tests based on CLI arguments."""
        logger.info(f"Running tests - Type: {args.test_type}, Module: {args.module}")
        
        # Build test suites
        suites = self.suite_builder.build_suites(args.test_type, args.module)
        
        if not suites:
            logger.error(f"No test suites found for type: {args.test_type}, module: {args.module}")
            return {"error": "No test suites found"}
        
        logger.info(f"Found {len(suites)} test suites")
        
        # Execute tests
        all_results = []
        
        for suite in suites:
            logger.info(f"Executing suite: {suite.name}")
            
            results = await self.executor.execute_test_suite(
                suite,
                parallel=args.parallel,
                timeout=args.timeout,
                coverage=args.coverage
            )
            
            all_results.extend(results)
            
            # Print immediate results
            passed = len([r for r in results if r.status.value == "passed"])
            failed = len([r for r in results if r.status.value == "failed"])
            print(f"Suite {suite.name}: {passed} passed, {failed} failed")
        
        # Generate report
        if args.format == "json":
            report = self.reporter.generate_json_report(all_results)
            print(report)
        else:
            report = self.reporter.generate_summary_report(all_results)
            print(report)
        
        # Save report if requested
        if args.output:
            report_file = self.reporter.save_report(all_results, args.format)
            logger.info(f"Report saved to: {report_file}")
        
        return {
            "total_tests": len(all_results),
            "passed": len([r for r in all_results if r.status.value == "passed"]),
            "failed": len([r for r in all_results if r.status.value == "failed"]),
            "suites": len(suites)
        }
    
    async def create_workflow(self, args: argparse.Namespace) -> str:
        """Create a test workflow."""
        logger.info(f"Creating workflow: {args.name}")
        
        priority = TestPriority(args.priority) if args.priority else TestPriority.MEDIUM
        
        workflow_id = await self.orchestrator.create_workflow(
            name=args.name,
            test_type=args.test_type,
            module=args.module,
            priority=priority,
            timeout=args.timeout,
            parallel=args.parallel,
            coverage=args.coverage
        )
        
        logger.info(f"Created workflow: {workflow_id}")
        return workflow_id
    
    async def execute_workflow(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Execute a test workflow."""
        logger.info(f"Executing workflow: {args.workflow_id}")
        
        workflow = await self.orchestrator.execute_workflow(args.workflow_id)
        
        # Print results
        status = self.orchestrator.get_workflow_status(args.workflow_id)
        print(json.dumps(status, indent=2))
        
        return status
    
    def discover_tests(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Discover available tests."""
        logger.info(f"Discovering tests - Type: {args.test_type}")
        
        suites = self.suite_builder.build_suites(args.test_type, "all")
        
        discovery_info = {
            "test_types": {},
            "total_suites": len(suites),
            "total_tests": 0
        }
        
        for suite in suites:
            if suite.type not in discovery_info["test_types"]:
                discovery_info["test_types"][suite.type] = {
                    "suites": 0,
                    "tests": 0,
                    "modules": set()
                }
            
            type_info = discovery_info["test_types"][suite.type]
            type_info["suites"] += 1
            type_info["tests"] += len(suite.tests)
            type_info["modules"].add(suite.module)
            discovery_info["total_tests"] += len(suite.tests)
        
        # Convert sets to lists for JSON serialization
        for type_info in discovery_info["test_types"].values():
            type_info["modules"] = list(type_info["modules"])
        
        if args.format == "json":
            print(json.dumps(discovery_info, indent=2))
        else:
            print(f"Test Discovery Results")
            print("=" * 50)
            print(f"Total Suites: {discovery_info['total_suites']}")
            print(f"Total Tests: {discovery_info['total_tests']}")
            print()
            
            for test_type, info in discovery_info["test_types"].items():
                print(f"{test_type.upper()}: {info['suites']} suites, {info['tests']} tests")
                print(f"  Modules: {', '.join(info['modules'])}")
                print()
        
        return discovery_info
    
    def get_agent_status(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Get status of test agents."""
        status = self.orchestrator.get_agent_status()
        
        if args.format == "json":
            print(json.dumps(status, indent=2))
        else:
            print("Agent Status")
            print("=" * 50)
            for agent_id, agent_info in status.items():
                print(f"{agent_info['name']} ({agent_id})")
                print(f"  Status: {agent_info['status']}")
                print(f"  Load: {agent_info['current_load']}/{agent_info['max_concurrent']}")
                print(f"  Capabilities: {', '.join(agent_info['capabilities'])}")
                if agent_info['assigned_workflows']:
                    print(f"  Workflows: {', '.join(agent_info['assigned_workflows'])}")
                print()
        
        return status

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="A2A Test MCP CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run unit tests for a2aAgents module
  python cli.py run --test-type unit --module a2aAgents --coverage
  
  # Discover all available tests
  python cli.py discover --test-type all
  
  # Create and execute a workflow
  python cli.py workflow create --name "CI Tests" --test-type unit --priority high
  
  # Get agent status
  python cli.py agents status
        """
    )
    
    parser.add_argument(
        "--test-root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Test root directory (default: ../)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run tests command
    run_parser = subparsers.add_parser("run", help="Execute tests")
    run_parser.add_argument("--test-type", choices=["unit", "integration", "e2e", "performance", "security", "contracts", "all"], default="unit")
    run_parser.add_argument("--module", choices=["a2aAgents", "a2aNetwork", "common", "all"], default="all")
    run_parser.add_argument("--parallel", action="store_true", help="Enable parallel execution")
    run_parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    run_parser.add_argument("--timeout", type=int, default=300, help="Test timeout in seconds")
    run_parser.add_argument("--format", choices=["text", "json"], default="text")
    run_parser.add_argument("--output", action="store_true", help="Save report to file")
    
    # Discover tests command
    discover_parser = subparsers.add_parser("discover", help="Discover available tests")
    discover_parser.add_argument("--test-type", choices=["unit", "integration", "e2e", "performance", "security", "contracts", "all"], default="all")
    discover_parser.add_argument("--format", choices=["text", "json"], default="text")
    
    # Workflow commands
    workflow_parser = subparsers.add_parser("workflow", help="Workflow management")
    workflow_subparsers = workflow_parser.add_subparsers(dest="workflow_action")
    
    # Create workflow
    create_parser = workflow_subparsers.add_parser("create", help="Create workflow")
    create_parser.add_argument("--name", required=True, help="Workflow name")
    create_parser.add_argument("--test-type", choices=["unit", "integration", "e2e", "performance", "security", "contracts", "all"], default="unit")
    create_parser.add_argument("--module", choices=["a2aAgents", "a2aNetwork", "common", "all"], default="all")
    create_parser.add_argument("--priority", choices=["critical", "high", "medium", "low"], default="medium")
    create_parser.add_argument("--parallel", action="store_true", help="Enable parallel execution")
    create_parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    create_parser.add_argument("--timeout", type=int, default=600, help="Workflow timeout in seconds")
    
    # Execute workflow
    execute_parser = workflow_subparsers.add_parser("execute", help="Execute workflow")
    execute_parser.add_argument("workflow_id", help="Workflow ID to execute")
    
    # Agent commands
    agents_parser = subparsers.add_parser("agents", help="Agent management")
    agents_subparsers = agents_parser.add_subparsers(dest="agents_action")
    
    # Agent status
    status_parser = agents_subparsers.add_parser("status", help="Get agent status")
    status_parser.add_argument("--format", choices=["text", "json"], default="text")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = A2ATestCLI(args.test_root)
    
    # Execute command
    try:
        if args.command == "run":
            result = asyncio.run(cli.run_tests(args))
            
        elif args.command == "discover":
            result = cli.discover_tests(args)
            
        elif args.command == "workflow":
            if args.workflow_action == "create":
                workflow_id = asyncio.run(cli.create_workflow(args))
                print(f"Created workflow: {workflow_id}")
            elif args.workflow_action == "execute":
                result = asyncio.run(cli.execute_workflow(args))
            else:
                workflow_parser.print_help()
                
        elif args.command == "agents":
            if args.agents_action == "status":
                result = cli.get_agent_status(args)
            else:
                agents_parser.print_help()
        
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()