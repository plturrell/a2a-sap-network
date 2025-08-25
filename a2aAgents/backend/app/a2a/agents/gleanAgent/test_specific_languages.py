#!/usr/bin/env python3
"""
Test specific language support in GleanAgent on real project files
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directories to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
backend_dir = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(backend_dir))

# Set required environment variables
os.environ["A2A_SERVICE_URL"] = "http://localhost:3000"
os.environ["A2A_SERVICE_HOST"] = "localhost"
os.environ["A2A_BASE_URL"] = "http://localhost:3000"
os.environ["BLOCKCHAIN_ENABLED"] = "false"

from gleanAgentSdk import GleanAgent


async def test_specific_file_types():
    """Test specific file types in the project"""
    agent = GleanAgent()

    project_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent

    # Find and analyze specific file types
    file_types_to_test = {
        "YAML": {"patterns": ["*.yaml", "*.yml"], "examples": []},
        "JSON": {"patterns": ["*.json"], "examples": []},
        "HTML": {"patterns": ["*.html"], "examples": []},
        "XML": {"patterns": ["*.xml"], "examples": []},
        "Shell": {"patterns": ["*.sh"], "examples": []},
        "CSS": {"patterns": ["*.css"], "examples": []},
        "SCSS": {"patterns": ["*.scss", "*.sass"], "examples": []},
    }

    # Find example files for each type
    print("🔍 Finding files by type in the project...")
    for lang, info in file_types_to_test.items():
        for pattern in info["patterns"]:
            files = list(project_root.rglob(pattern))[:3]  # Get up to 3 examples
            info["examples"].extend(files)
        print(f"{lang}: Found {len(info['examples'])} files")

    # Test each file type
    print("\n🧪 Testing Language-Specific Linters:")
    print("=" * 70)

    for lang, info in file_types_to_test.items():
        if info["examples"]:
            print(f"\n📄 Testing {lang} files:")

            # Get the appropriate linter method
            method_name = f"_run_{lang.lower()}_linters_batch"
            if hasattr(agent, method_name):
                try:
                    result = await getattr(agent, method_name)(info["examples"], str(project_root))
                    issues = result.get('issues', [])
                    linter_results = result.get('linter_results', {})

                    print(f"  ✓ Linter executed successfully")
                    print(f"  📊 Issues found: {len(issues)}")

                    # Show linter results
                    if linter_results:
                        print(f"  🛠️  Linter results:")
                        for linter, status in linter_results.items():
                            if "Error" in str(status):
                                print(f"    ❌ {linter}: {status}")
                            else:
                                print(f"    ✅ {linter}: Success")

                    # Show sample issues
                    if issues:
                        print(f"  ⚠️  Sample issues (up to 3):")
                        for issue in issues[:3]:
                            file_name = Path(issue.get('file_path', '')).name
                            message = issue.get('message', 'No message')
                            line = issue.get('line', '?')
                            severity = issue.get('severity', 'unknown')
                            tool = issue.get('tool', 'unknown')

                            print(f"    [{severity}] {file_name}:{line}")
                            print(f"      {message} (via {tool})")

                    # Show example files analyzed
                    print(f"  📁 Example files analyzed:")
                    for file in info["examples"][:3]:
                        rel_path = file.relative_to(project_root)
                        print(f"    - {rel_path}")

                except Exception as e:
                    print(f"  ❌ Error: {str(e)}")
            else:
                print(f"  ❌ Method {method_name} not found")


async def analyze_specific_directories():
    """Analyze directories known to contain specific file types"""
    agent = GleanAgent()

    project_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent

    # Specific directories with known file types
    test_locations = [
        {
            "path": project_root / ".github" / "workflows",
            "description": "GitHub Actions YAML files",
            "expected_types": ["yaml"]
        },
        {
            "path": project_root / "scripts",
            "description": "Shell scripts",
            "expected_types": ["shell"]
        },
        {
            "path": project_root / "mta_archives",
            "description": "MTA deployment descriptors",
            "expected_types": ["yaml", "json"]
        },
        {
            "path": project_root / "src",
            "description": "Frontend source with CSS/HTML",
            "expected_types": ["css", "scss", "html"]
        }
    ]

    print("\n📁 Analyzing Specific Directories:")
    print("=" * 70)

    for location in test_locations:
        if location["path"].exists():
            print(f"\n🔍 Analyzing: {location['path'].relative_to(project_root)}")
            print(f"   Description: {location['description']}")

            # Run comprehensive analysis on the directory
            result = await agent.analyze_code_comprehensive(str(location["path"]))

            if result.get("success"):
                print(f"   ✅ Analysis successful")
                print(f"   📊 Files analyzed: {result.get('files_analyzed', 0)}")
                print(f"   ⚠️  Total issues: {result.get('total_issues', 0)}")

                # Check which file types were found
                issues = result.get("issues", [])
                if issues:
                    file_types = set()
                    for issue in issues:
                        ext = Path(issue.get('file_path', '')).suffix.lower()
                        if ext:
                            file_types.add(ext)

                    print(f"   📄 File types with issues: {', '.join(sorted(file_types))}")

                    # Show breakdown by severity
                    severity_counts = result.get("issues_by_severity", {})
                    if severity_counts:
                        print(f"   🔴 By severity: {severity_counts}")
            else:
                print(f"   ❌ Analysis failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"\n❌ Directory not found: {location['path'].relative_to(project_root)}")


async def main():
    """Run focused tests on new language support"""
    print("🚀 Testing Extended Language Support on Real Project Files\n")

    # Test 1: Test specific file types
    await test_specific_file_types()

    # Test 2: Analyze specific directories
    await analyze_specific_directories()

    print("\n✅ Testing completed!")
    print("\n💡 Note: Install these linters for better results:")
    print("   - htmlhint: npm install -g htmlhint")
    print("   - yamllint: pip install yamllint")
    print("   - shellcheck: brew install shellcheck")
    print("   - stylelint: npm install -g stylelint")
    print("   - xmllint: Usually pre-installed on Unix systems")


if __name__ == "__main__":
    asyncio.run(main())