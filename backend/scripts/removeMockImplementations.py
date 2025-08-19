#!/usr/bin/env python3
"""
Script to identify and remove mock/fallback implementations from A2A agents.
"""
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class MockRemover:
    """Identifies and helps remove mock implementations."""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.issues_found = []
        self.files_fixed = []
        
        # Patterns to search for
        self.mock_patterns = [
            r'mock',
            r'fallback',
            r'TODO',
            r'FIXME',
            r'dev_fallback',
            r'placeholder',
            r'dummy',
            r'fake',
            r'stub',
            r'test_only',
            r'not suitable for production',
            r'temporary implementation',
            r'hardcoded',
            r'localhost:\d+',
            r'127\.0\.0\.1',
            r'0x0{40}',  # Zero addresses
            r'/tmp/',
        ]
        
        # Agent directories to check
        self.agent_dirs = [
            "app/a2a/agents/agentManager",
            "app/a2a/agents/agent0DataProduct", 
            "app/a2a/agents/agent1Standardization",
            "app/a2a/agents/agent2AiPreparation",
            "app/a2a/agents/agent3VectorProcessing",
            "app/a2a/agents/agent4CalcValidation",
            "app/a2a/agents/agent5QaValidation",
            "app/a2a/agents/dataManager",
            "app/a2a/agents/catalogManager",
            "app/a2a/agents/agentBuilder",
            "app/a2a/agents/calculationAgent",
            "app/a2a/agents/reasoningAgent",
            "app/a2a/agents/sqlAgent",
        ]
    
    def scan_file(self, file_path: Path) -> List[Tuple[int, str, str]]:
        """
        Scan a file for mock implementations.
        
        Returns:
            List of (line_number, pattern_matched, line_content)
        """
        issues = []
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines, 1):
                for pattern in self.mock_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Skip comments and imports
                        stripped = line.strip()
                        if stripped.startswith('#') or stripped.startswith('//'):
                            continue
                        if 'import' in line and 'mock' in line.lower():
                            continue
                        
                        issues.append((i, pattern, line.rstrip()))
                        break
        
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
        
        return issues
    
    def scan_all_agents(self):
        """Scan all agent files for mock implementations."""
        print("Scanning A2A agents for mock/fallback implementations...")
        print("=" * 80)
        
        total_issues = 0
        
        for agent_dir in self.agent_dirs:
            agent_path = self.base_path / agent_dir
            if not agent_path.exists():
                continue
            
            agent_issues = []
            
            # Scan all Python files in agent directory
            for py_file in agent_path.rglob("*.py"):
                # Skip test files
                if 'test' in str(py_file).lower():
                    continue
                
                file_issues = self.scan_file(py_file)
                if file_issues:
                    agent_issues.append((py_file, file_issues))
            
            if agent_issues:
                print(f"\n{agent_dir}:")
                for file_path, issues in agent_issues:
                    print(f"  {file_path.relative_to(self.base_path)}:")
                    for line_num, pattern, content in issues:
                        print(f"    Line {line_num}: [{pattern}] {content[:80]}...")
                        total_issues += 1
                        self.issues_found.append({
                            'file': str(file_path),
                            'line': line_num,
                            'pattern': pattern,
                            'content': content
                        })
        
        print(f"\n{'-' * 80}")
        print(f"Total issues found: {total_issues}")
        return total_issues
    
    def generate_fixes(self):
        """Generate fix recommendations for found issues."""
        print("\n" + "=" * 80)
        print("FIX RECOMMENDATIONS")
        print("=" * 80)
        
        fixes = {
            'mock': "Replace with actual implementation",
            'fallback': "Remove fallback and use production implementation",
            'TODO': "Complete the TODO implementation",
            'FIXME': "Fix the identified issue",
            'localhost': "Use configuration from agentConfig.py",
            '127.0.0.1': "Use configuration from agentConfig.py",
            '0x0{40}': "Use actual deployed contract address",
            '/tmp/': "Use configured storage path from agentConfig.py",
            'hardcoded': "Move to environment configuration",
            'placeholder': "Replace with actual value",
            'not suitable for production': "Replace with production-ready code"
        }
        
        recommendations = {}
        
        for issue in self.issues_found:
            file_path = issue['file']
            if file_path not in recommendations:
                recommendations[file_path] = []
            
            for fix_pattern, fix_action in fixes.items():
                if fix_pattern in issue['pattern'] or fix_pattern in issue['content'].lower():
                    recommendations[file_path].append({
                        'line': issue['line'],
                        'action': fix_action,
                        'content': issue['content']
                    })
                    break
        
        for file_path, file_fixes in recommendations.items():
            print(f"\n{Path(file_path).relative_to(self.base_path)}:")
            for fix in file_fixes:
                print(f"  Line {fix['line']}: {fix['action']}")
                print(f"    Current: {fix['content'][:60]}...")
    
    def check_imports(self):
        """Check for proper imports instead of fallbacks."""
        print("\n" + "=" * 80)
        print("IMPORT VALIDATION")
        print("=" * 80)
        
        required_imports = {
            'trustSystem.trustIntegration': ['sign_a2a_message', 'verify_a2a_message', 'initialize_agent_trust'],
            'config.agentConfig': ['config'],
            'common.errorHandling': ['CircuitBreaker', 'with_circuit_breaker', 'with_retry'],
        }
        
        for agent_dir in self.agent_dirs:
            agent_path = self.base_path / agent_dir
            if not agent_path.exists():
                continue
            
            main_file = None
            for pattern in ['*Agent.py', '*AgentSdk.py', 'active/*.py']:
                files = list(agent_path.glob(pattern))
                if files:
                    main_file = files[0]
                    break
            
            if not main_file:
                continue
            
            try:
                with open(main_file, 'r') as f:
                    content = f.read()
                
                missing_imports = []
                for module, items in required_imports.items():
                    if module not in content:
                        missing_imports.append(module)
                
                if missing_imports:
                    print(f"\n{agent_dir}: Missing imports:")
                    for imp in missing_imports:
                        print(f"  - from {imp} import ...")
            
            except Exception as e:
                print(f"Error checking {main_file}: {e}")
    
    def generate_report(self):
        """Generate a detailed report of findings."""
        report_path = self.base_path / "mock_implementation_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Mock Implementation Report\n\n")
            f.write(f"Generated on: {Path(__file__).name}\n\n")
            f.write(f"Total issues found: {len(self.issues_found)}\n\n")
            
            # Group by file
            by_file = {}
            for issue in self.issues_found:
                file_path = issue['file']
                if file_path not in by_file:
                    by_file[file_path] = []
                by_file[file_path].append(issue)
            
            f.write("## Issues by File\n\n")
            for file_path, issues in sorted(by_file.items()):
                rel_path = Path(file_path).relative_to(self.base_path)
                f.write(f"### {rel_path}\n\n")
                f.write(f"Issues found: {len(issues)}\n\n")
                
                for issue in issues:
                    f.write(f"- **Line {issue['line']}**: `{issue['pattern']}` pattern\n")
                    f.write(f"  ```python\n  {issue['content']}\n  ```\n\n")
            
            f.write("\n## Summary by Pattern\n\n")
            pattern_count = {}
            for issue in self.issues_found:
                pattern = issue['pattern']
                pattern_count[pattern] = pattern_count.get(pattern, 0) + 1
            
            for pattern, count in sorted(pattern_count.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- `{pattern}`: {count} occurrences\n")
        
        print(f"\nReport generated: {report_path}")


def main():
    """Run mock removal process."""
    remover = MockRemover()
    
    # Scan all agents
    total_issues = remover.scan_all_agents()
    
    if total_issues > 0:
        # Generate fixes
        remover.generate_fixes()
        
        # Check imports
        remover.check_imports()
        
        # Generate report
        remover.generate_report()
        
        print("\n" + "=" * 80)
        print("NEXT STEPS:")
        print("1. Review the mock_implementation_report.md")
        print("2. Apply the recommended fixes")
        print("3. Ensure all agents use:")
        print("   - trustSystem.trustIntegration for blockchain operations")
        print("   - config.agentConfig for configuration")
        print("   - common.errorHandling for resilience patterns")
        print("4. Run integration tests after fixes")
    else:
        print("\n✅ No mock implementations found!")
    
    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())