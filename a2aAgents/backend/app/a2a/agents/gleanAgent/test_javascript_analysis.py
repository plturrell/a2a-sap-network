#!/usr/bin/env python3
"""
Test JavaScript analysis capabilities of GleanAgentSdk on real JavaScript files
"""

import asyncio
import json
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from gleanAgentSdk import GleanAgentSdk


async def test_javascript_analysis():
    """Test JavaScript analysis on real files from the project"""

    # Initialize the SDK
    sdk = GleanAgentSdk(project_root="/Users/apple/projects/a2a")

    # JavaScript files to analyze
    js_files = [
        "/Users/apple/projects/a2a/a2aAgents/scripts/build/bookStandardization.js",
        "/Users/apple/projects/a2a/a2aAgents/scripts/build/accountStandardization.js",
        "/Users/apple/projects/a2a/a2aAgents/scripts/build/productStandardization.js"
    ]

    print("=" * 80)
    print("Testing JavaScript Analysis with GleanAgentSdk")
    print("=" * 80)

    for js_file in js_files:
        file_path = Path(js_file)
        if not file_path.exists():
            print(f"\nSkipping {js_file} - file not found")
            continue

        print(f"\nAnalyzing: {js_file}")
        print("-" * 80)

        # Get file info
        file_size = file_path.stat().st_size
        line_count = len(file_path.read_text().splitlines())
        print(f"File size: {file_size:,} bytes")
        print(f"Line count: {line_count:,} lines")

        # Run JavaScript linters batch
        print("\n1. Running JavaScript linters and analysis...")
        try:
            # Convert to list of Path objects for the method
            files_to_analyze = [file_path]
            directory = str(file_path.parent)

            # Call the JavaScript linters batch method
            result = await sdk._run_javascript_linters_batch(files_to_analyze, directory)

            # Display linter results
            print("\nLinter Results:")
            for linter, status in result.get("linter_results", {}).items():
                print(f"  - {linter}: {status}")

            # Display issues found
            issues = result.get("issues", [])
            if issues:
                print(f"\nTotal issues found: {len(issues)}")

                # Group issues by severity
                severity_counts = {}
                for issue in issues:
                    severity = issue.get("severity", "unknown")
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1

                print("\nIssues by severity:")
                for severity, count in sorted(severity_counts.items()):
                    print(f"  - {severity}: {count}")

                # Show first few issues as examples
                print("\nExample issues (showing first 5):")
                for i, issue in enumerate(issues[:5], 1):
                    print(f"\n  Issue {i}:")
                    print(f"    File: {issue.get('file_path', 'N/A')}")
                    print(f"    Line: {issue.get('line', 'N/A')}")
                    print(f"    Tool: {issue.get('tool', 'N/A')}")
                    print(f"    Severity: {issue.get('severity', 'N/A')}")
                    print(f"    Message: {issue.get('message', 'N/A')}")

                # Group issues by tool
                tool_counts = {}
                for issue in issues:
                    tool = issue.get("tool", "unknown")
                    tool_counts[tool] = tool_counts.get(tool, 0) + 1

                print("\nIssues by tool:")
                for tool, count in sorted(tool_counts.items()):
                    print(f"  - {tool}: {count}")
            else:
                print("\nNo issues found!")

        except Exception as e:
            print(f"Error during JavaScript analysis: {e}")
            import traceback
            traceback.print_exc()

        # Test individual analysis methods
        print("\n2. Testing semantic analysis...")
        try:
            semantic_result = await sdk._analyze_javascript_semantics([file_path])
            semantic_issues = semantic_result.get("issues", [])
            print(f"Semantic issues found: {len(semantic_issues)}")

            # Show semantic issue types
            if semantic_issues:
                semantic_types = {}
                for issue in semantic_issues:
                    msg = issue.get("message", "")
                    # Extract issue type from message
                    if "var" in msg:
                        issue_type = "var usage"
                    elif "===" in msg:
                        issue_type = "loose equality"
                    elif "semicolon" in msg:
                        issue_type = "missing semicolon"
                    elif "callback" in msg:
                        issue_type = "callback complexity"
                    elif "never used" in msg:
                        issue_type = "unused variable"
                    elif "arrow function" in msg:
                        issue_type = "function style"
                    elif "template literal" in msg:
                        issue_type = "string concatenation"
                    elif "Promise" in msg:
                        issue_type = "promise error handling"
                    else:
                        issue_type = "other"

                    semantic_types[issue_type] = semantic_types.get(issue_type, 0) + 1

                print("\nSemantic issue types:")
                for issue_type, count in sorted(semantic_types.items()):
                    print(f"  - {issue_type}: {count}")

        except Exception as e:
            print(f"Error during semantic analysis: {e}")

        print("\n3. Testing security analysis...")
        try:
            security_result = await sdk._analyze_javascript_security([file_path])
            security_issues = security_result.get("issues", [])
            print(f"Security issues found: {len(security_issues)}")

            if security_issues:
                # Show first few security issues
                print("\nExample security issues (showing first 3):")
                for i, issue in enumerate(security_issues[:3], 1):
                    print(f"\n  Security Issue {i}:")
                    print(f"    Line: {issue.get('line', 'N/A')}")
                    print(f"    Severity: {issue.get('severity', 'N/A')}")
                    print(f"    Message: {issue.get('message', 'N/A')}")

        except Exception as e:
            print(f"Error during security analysis: {e}")

        print("\n4. Testing performance analysis...")
        try:
            performance_result = await sdk._analyze_javascript_performance([file_path])
            performance_issues = performance_result.get("issues", [])
            print(f"Performance issues found: {len(performance_issues)}")

            if performance_issues:
                # Show performance issue types
                perf_types = {}
                for issue in performance_issues:
                    msg = issue.get("message", "")
                    if "loop" in msg.lower():
                        perf_type = "loop optimization"
                    elif "dom" in msg.lower():
                        perf_type = "DOM manipulation"
                    elif "memory" in msg.lower():
                        perf_type = "memory usage"
                    else:
                        perf_type = "other"

                    perf_types[perf_type] = perf_types.get(perf_type, 0) + 1

                print("\nPerformance issue types:")
                for perf_type, count in sorted(perf_types.items()):
                    print(f"  - {perf_type}: {count}")

        except Exception as e:
            print(f"Error during performance analysis: {e}")

        print("\n" + "-" * 80)

    # Test code structure analysis
    print("\n\nTesting code structure analysis...")
    print("=" * 80)

    try:
        # Analyze the standardization scripts directory
        scripts_dir = "/Users/apple/projects/a2a/a2aAgents/scripts/build"
        structure_result = await sdk.analyze_code_structure(scripts_dir)

        print(f"\nCode structure analysis for: {scripts_dir}")

        if "summary" in structure_result:
            summary = structure_result["summary"]
            print(f"\nProject type: {summary.get('project_type', 'Unknown')}")
            print(f"Total files: {summary.get('total_files', 0)}")
            print(f"Total lines: {summary.get('total_lines', 0)}")

            if "languages" in summary:
                print("\nLanguage distribution:")
                for lang, count in summary["languages"].items():
                    if count > 0:
                        print(f"  - {lang}: {count} files")

        if "dependencies" in structure_result:
            deps = structure_result["dependencies"]
            if deps.get("javascript", {}).get("npm", []):
                print("\nJavaScript dependencies found:")
                for dep in deps["javascript"]["npm"][:5]:  # Show first 5
                    print(f"  - {dep}")

    except Exception as e:
        print(f"Error during code structure analysis: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("JavaScript analysis testing completed!")
    print("=" * 80)


if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_javascript_analysis())
