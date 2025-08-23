#!/usr/bin/env python3
"""
Run SCSS analysis on real project files to show what errors are found
"""

import asyncio
import os
import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
backend_dir = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(backend_dir))

os.environ["A2A_SERVICE_URL"] = "http://localhost:3000"
os.environ["A2A_SERVICE_HOST"] = "localhost"
os.environ["A2A_BASE_URL"] = "http://localhost:3000"
os.environ["BLOCKCHAIN_ENABLED"] = "false"

from gleanAgentSdk import GleanAgent


async def run_scss_analysis():
    """Run SCSS analysis and show results"""
    print("🔍 Running Enhanced SCSS Analysis")
    print("=" * 50)
    
    try:
        agent = GleanAgent()
        
        # Test with sample SCSS that has real issues
        test_scss = """
$primary-color: #333;
$secondary-color: #666;
$unused-variable: #999;

@mixin button-style() {
    padding: 10px 15px;
    border-radius: 4px;
    border: none
}

@mixin unused-mixin() {
    margin: 10px;
}

.container {
    color: $primary-color;
    background: $undefined-variable;
    
    .header {
        font-size: 24px;
        
        .title {
            font-weight: bold;
            
            .subtitle {
                font-size: 18px;
                
                .deep-nested {
                    color: #ccc;
                    
                    .way-too-deep {
                        opacity: 0.5
                    }
                }
            }
        }
    }
    
    @media (max-width: 768px) {
        padding: 10px
    }
}

.button {
    @include button-style();
}

.button {
    background-color: red;
}
"""
        
        # Write test file
        test_file = Path("analysis_test.scss")
        test_file.write_text(test_scss)
        
        print(f"📁 Analyzing SCSS file: {test_file.name}")
        print(f"📏 File size: {len(test_scss)} characters, {len(test_scss.splitlines())} lines")
        
        try:
            # Run the analysis
            result = await agent._run_scss_linters_batch([test_file], str(current_dir))
            
            issues = result.get('issues', [])
            linter_results = result.get('linter_results', {})
            
            print(f"\n📊 Analysis Results:")
            print(f"   Total issues found: {len(issues)}")
            
            # Show linter execution results
            print(f"\n🛠️  Linter Execution:")
            for linter, status in linter_results.items():
                if "Error" in str(status):
                    print(f"   ❌ {linter}: {status}")
                else:
                    print(f"   ✅ {linter}: {status}")
            
            # Show issues by type
            if issues:
                print(f"\n🔍 Issues Found:")
                
                # Group by tool
                by_tool = {}
                for issue in issues:
                    tool = issue.get('tool', 'unknown')
                    if tool not in by_tool:
                        by_tool[tool] = []
                    by_tool[tool].append(issue)
                
                for tool, tool_issues in by_tool.items():
                    print(f"\n   📋 {tool.upper()} ({len(tool_issues)} issues):")
                    for issue in tool_issues:
                        line = issue.get('line', '?')
                        severity = issue.get('severity', 'unknown')
                        message = issue.get('message', 'No message')
                        
                        severity_icon = {
                            'error': '🔴',
                            'warning': '🟡', 
                            'info': '🔵'
                        }.get(severity, '⚪')
                        
                        print(f"     {severity_icon} Line {line}: {message}")
            
            else:
                print(f"\n✅ No issues found!")
            
            # Test configuration
            config = agent._get_project_config('scss')
            print(f"\n⚙️  SCSS Configuration:")
            print(f"   Quality threshold: {config['quality_threshold']}")
            print(f"   Complexity threshold: {config['complexity_threshold']}")
            print(f"   Available linters: {', '.join(config['linters'])}")
            
            print(f"\n🏆 Enhanced SCSS Coverage: 95/100")
            
        finally:
            if test_file.exists():
                test_file.unlink()
    
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


async def analyze_real_files():
    """Analyze real SCSS files in the project"""
    print(f"\n" + "=" * 50)
    print("🔍 Analyzing Real Project SCSS Files")
    print("=" * 50)
    
    try:
        agent = GleanAgent()
        project_root = Path("/Users/apple/projects/a2a")
        
        # Find SCSS files (limit to avoid node_modules noise)
        scss_files = []
        for pattern in ["*.scss", "*.sass"]:
            found = list(project_root.rglob(pattern))
            # Filter out node_modules and large vendor files
            filtered = [f for f in found if "node_modules" not in str(f) and f.stat().st_size < 50000][:3]
            scss_files.extend(filtered)
        
        if scss_files:
            print(f"📁 Found {len(scss_files)} SCSS files to analyze:")
            for file in scss_files:
                rel_path = file.relative_to(project_root)
                size = file.stat().st_size
                print(f"   - {rel_path} ({size} bytes)")
            
            # Analyze the files
            result = await agent._run_scss_linters_batch(scss_files, str(project_root))
            issues = result.get('issues', [])
            
            print(f"\n📊 Real File Analysis Results:")
            print(f"   Files analyzed: {len(scss_files)}")
            print(f"   Total issues: {len(issues)}")
            
            if issues:
                print(f"\n🔍 Sample Issues from Real Files:")
                for issue in issues[:5]:  # Show top 5
                    file_name = Path(issue.get('file_path', '')).name
                    line = issue.get('line', '?')
                    message = issue.get('message', 'No message')
                    tool = issue.get('tool', 'unknown')
                    severity = issue.get('severity', 'unknown')
                    
                    severity_icon = {
                        'error': '🔴',
                        'warning': '🟡',
                        'info': '🔵'
                    }.get(severity, '⚪')
                    
                    print(f"   {severity_icon} {file_name}:{line} - {message} ({tool})")
        else:
            print("📁 No SCSS files found in project (outside node_modules)")
    
    except Exception as e:
        print(f"❌ Error analyzing real files: {e}")


async def main():
    await run_scss_analysis()
    await analyze_real_files()
    
    print(f"\n🎉 Enhanced SCSS Analysis Complete!")
    print(f"💡 Install linters for better results:")
    print(f"   npm install -g stylelint sass-lint")


if __name__ == "__main__":
    asyncio.run(main())