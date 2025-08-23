#!/usr/bin/env python3
"""
Test script for validating extended language support in GleanAgent on the real A2A project
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directories to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
# Add backend directory to path for app.a2a imports
backend_dir = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(backend_dir))

from gleanAgentSdk import GleanAgent


async def analyze_real_project():
    """Analyze the real A2A project codebase"""
    agent = GleanAgent()
    
    # Get the root project directory (5 levels up from this script)
    project_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
    print(f"🔍 Analyzing project at: {project_root}")
    
    # First, detect the project type
    project_type = agent._detect_project_type(str(project_root))
    print(f"\n📁 Detected project type: {project_type}")
    
    # Configure agent for the project
    config_result = await agent.configure_for_project(str(project_root))
    if config_result.get("success"):
        print(f"✅ Configured for {config_result['project_type']} project")
    
    print("\n🔧 Running Comprehensive Analysis on Real Project...")
    print("=" * 70)
    
    # Run the analysis
    result = await agent.analyze_code_comprehensive(str(project_root))
    
    if result.get("success"):
        print(f"\n✅ Analysis completed successfully!")
        print(f"\n📊 Analysis Summary:")
        print(f"  - Analysis ID: {result.get('analysis_id', 'N/A')}")
        print(f"  - Files analyzed: {result.get('files_analyzed', 0)}")
        print(f"  - Total issues: {result.get('total_issues', 0)}")
        print(f"  - Duration: {result.get('duration', 0):.2f}s")
        
        # Show file type breakdown
        print("\n📁 Files by Language:")
        # Count files by extension
        file_counts = {}
        analyzed_files = result.get('files', [])
        if not analyzed_files and project_root.exists():
            # If no files info in result, scan directory
            for ext in ['*.py', '*.js', '*.ts', '*.html', '*.xml', '*.yaml', '*.yml', 
                       '*.json', '*.sh', '*.css', '*.scss', '*.sol', '*.cds']:
                count = len(list(project_root.rglob(ext)))
                if count > 0:
                    file_counts[ext] = count
        
        for ext, count in sorted(file_counts.items()):
            print(f"    {ext}: {count} files")
        
        # Show issues by severity
        severity_counts = result.get("issues_by_severity", {})
        if severity_counts:
            print("\n⚠️  Issues by Severity:")
            for severity, count in sorted(severity_counts.items()):
                print(f"    {severity}: {count}")
        
        # Show issues by type
        type_counts = result.get("issues_by_type", {})
        if type_counts:
            print("\n📝 Issues by Type:")
            for issue_type, count in sorted(type_counts.items())[:10]:
                print(f"    {issue_type}: {count}")
        
        # Show sample issues for each language
        issues = result.get("issues", [])
        if issues:
            print("\n🔍 Sample Issues by Language:")
            
            # Group issues by file extension
            issues_by_lang = {}
            for issue in issues:
                file_path = issue.get('file_path', '')
                ext = Path(file_path).suffix.lower()
                if ext not in issues_by_lang:
                    issues_by_lang[ext] = []
                issues_by_lang[ext].append(issue)
            
            # Show samples for each language
            for ext, lang_issues in sorted(issues_by_lang.items()):
                lang_name = {
                    '.py': 'Python',
                    '.js': 'JavaScript',
                    '.ts': 'TypeScript',
                    '.html': 'HTML',
                    '.xml': 'XML',
                    '.yaml': 'YAML',
                    '.yml': 'YAML',
                    '.json': 'JSON',
                    '.sh': 'Shell',
                    '.css': 'CSS',
                    '.scss': 'SCSS'
                }.get(ext, ext.upper())
                
                print(f"\n  {lang_name} Issues (showing up to 3):")
                for issue in lang_issues[:3]:
                    severity = issue.get('severity', 'unknown')
                    tool = issue.get('tool', 'unknown')
                    message = issue.get('message', 'No message')
                    file_path = issue.get('file_path', 'unknown')
                    line = issue.get('line', '?')
                    
                    # Truncate long file paths
                    if len(file_path) > 50:
                        file_path = "..." + file_path[-47:]
                    
                    print(f"    [{severity}] {file_path}:{line} - {message} (via {tool})")
        
        # Show linter results summary
        linter_results = result.get("linter_results", {})
        if linter_results:
            print("\n🛠️  Linter Execution Summary:")
            for linter, status in sorted(linter_results.items()):
                if "Error" not in str(status):
                    print(f"    ✓ {linter}: Success")
                else:
                    print(f"    ✗ {linter}: {status}")
    else:
        print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}")
        if 'details' in result:
            print(f"Details: {result['details']}")


async def analyze_specific_directories():
    """Analyze specific directories with different file types"""
    agent = GleanAgent()
    
    project_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
    
    # Specific directories to analyze
    test_dirs = [
        (project_root / "a2aNetwork", "Backend with Python, YAML, JSON"),
        (project_root / "docs", "Documentation with HTML, Markdown"),
        (project_root / "scripts", "Shell scripts"),
        (project_root / ".github/workflows", "GitHub Actions YAML files"),
        (project_root / "src", "Frontend with CSS/SCSS")
    ]
    
    print("\n🔍 Analyzing Specific Directories:")
    print("=" * 70)
    
    for dir_path, description in test_dirs:
        if dir_path.exists():
            print(f"\n📁 Analyzing: {dir_path.name} ({description})")
            result = await agent.analyze_code_comprehensive(str(dir_path))
            
            if result.get("success"):
                print(f"  ✓ Files analyzed: {result.get('files_analyzed', 0)}")
                print(f"  ✓ Issues found: {result.get('total_issues', 0)}")
                
                # Show file types found
                issues = result.get("issues", [])
                if issues:
                    file_types = set()
                    for issue in issues:
                        ext = Path(issue.get('file_path', '')).suffix.lower()
                        if ext:
                            file_types.add(ext)
                    if file_types:
                        print(f"  ✓ File types with issues: {', '.join(sorted(file_types))}")
            else:
                print(f"  ✗ Analysis failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"\n📁 Directory not found: {dir_path}")


async def test_language_specific_features():
    """Test language-specific features"""
    agent = GleanAgent()
    project_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
    
    print("\n🧪 Testing Language-Specific Features:")
    print("=" * 70)
    
    # Find sample files for each language
    language_samples = {
        "HTML": list(project_root.rglob("*.html"))[:2],
        "XML": list(project_root.rglob("*.xml"))[:2],
        "YAML": list(project_root.rglob("*.yaml"))[:2] + list(project_root.rglob("*.yml"))[:2],
        "JSON": list(project_root.rglob("*.json"))[:2],
        "Shell": list(project_root.rglob("*.sh"))[:2],
        "CSS": list(project_root.rglob("*.css"))[:2],
        "SCSS": list(project_root.rglob("*.scss"))[:2]
    }
    
    for lang, files in language_samples.items():
        if files:
            print(f"\n{lang} Files Found ({len(files)} samples):")
            for file in files:
                rel_path = file.relative_to(project_root)
                print(f"  - {rel_path}")
    
    # Test individual linter methods
    print("\n🔧 Testing Individual Linter Methods:")
    
    for lang, files in language_samples.items():
        if files:
            print(f"\n{lang} Linting:")
            method_name = f"_run_{lang.lower()}_linters_batch"
            if hasattr(agent, method_name):
                try:
                    result = await getattr(agent, method_name)(files, str(project_root))
                    issues = result.get('issues', [])
                    print(f"  ✓ Method exists and executed")
                    print(f"  ✓ Issues found: {len(issues)}")
                    if issues:
                        print(f"  ✓ Sample issue: {issues[0].get('message', 'N/A')[:80]}...")
                except Exception as e:
                    print(f"  ✗ Error: {str(e)}")
            else:
                print(f"  ✗ Method {method_name} not found")


async def main():
    """Run all tests on the real project"""
    print("🚀 Testing Extended Language Support on Real A2A Project\n")
    
    # Test 1: Comprehensive analysis of the entire project
    await analyze_real_project()
    
    # Test 2: Analyze specific directories
    await analyze_specific_directories()
    
    # Test 3: Test language-specific features
    await test_language_specific_features()
    
    print("\n✅ All tests completed!")
    print("\n💡 Note: Some linters may not be installed on your system.")
    print("   Install them for better analysis:")
    print("   - HTML: npm install -g htmlhint")
    print("   - YAML: pip install yamllint")
    print("   - Shell: brew install shellcheck (macOS) or apt-get install shellcheck (Linux)")
    print("   - CSS/SCSS: npm install -g stylelint sass-lint")


if __name__ == "__main__":
    asyncio.run(main())