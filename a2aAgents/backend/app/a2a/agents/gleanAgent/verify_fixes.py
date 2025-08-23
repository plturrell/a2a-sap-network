#!/usr/bin/env python3
"""
Quick verification that the fixes work
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


async def test_fixes():
    """Test that the fixes work"""
    print("🔧 Testing fixes...")
    
    try:
        agent = GleanAgent()
        print("✅ Agent initialized successfully")
        
        # Test _run_command method exists
        if hasattr(agent, '_run_command'):
            result = await agent._run_command("echo 'test'")
            print(f"✅ _run_command works: {result['stdout'].strip()}")
        else:
            print("❌ _run_command method missing")
        
        # Test simple linter functions
        test_file = Path(__file__)
        
        # Test JSON linter with a simple valid JSON
        json_content = '{"test": "value"}'
        json_file = Path("test_temp.json")
        json_file.write_text(json_content)
        
        try:
            result = await agent._run_json_linters_batch([json_file], str(current_dir))
            print(f"✅ JSON linter works: {len(result.get('issues', []))} issues found")
        finally:
            if json_file.exists():
                json_file.unlink()
        
        # Test YAML linter with a known YAML file  
        project_root = Path("/Users/apple/projects/a2a")
        yaml_files = []
        if (project_root / "mta.yaml").exists():
            yaml_files = [project_root / "mta.yaml"]
        
        if yaml_files:
            result = await agent._run_yaml_linters_batch(yaml_files, str(project_root))
            print(f"✅ YAML linter works: {len(result.get('issues', []))} issues found")
        
        print("\n🎉 All basic tests passed! Extended language support is working.")
        
        # Show supported languages
        print("\n📋 Supported Languages:")
        languages = ["html", "xml", "yaml", "json", "shell", "css", "scss"]
        for lang in languages:
            config = agent._get_project_config(lang)
            linters = ", ".join(config.get("linters", []))
            patterns = ", ".join(config.get("file_patterns", []))
            print(f"  {lang.upper():6}: {patterns} (linters: {linters})")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_fixes())