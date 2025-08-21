#!/usr/bin/env python3
"""
Quick setup test for A2A ChatAgent CLI
Validates that all dependencies are available and basic functionality works
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test that all required imports are available"""
    required_modules = [
        'asyncio',
        'argparse', 
        'json',
        'logging',
        'os',
        'time',
        'pathlib',
        'typing',
        'datetime'
    ]
    
    optional_modules = [
        'aiohttp',
        'yaml',
        'rich',
    ]
    
    print("🧪 Testing required imports...")
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            return False
    
    print("\n🧪 Testing optional imports...")
    
    missing_optional = []
    for module in optional_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"⚠️  {module}: {e}")
            missing_optional.append(module)
    
    if missing_optional:
        print(f"\n📦 Install missing dependencies: pip install {' '.join(missing_optional)}")
        print("📦 Or install all: pip install -r requirements-cli.txt")
    
    return True

def test_config_files():
    """Test that configuration files exist"""
    print("\n🧪 Testing configuration files...")
    
    config_dir = Path(__file__).parent / "config-examples"
    required_configs = ["development.yaml", "production.yaml", "testing.yaml"]
    
    if not config_dir.exists():
        print("❌ config-examples directory not found")
        return False
    
    for config in required_configs:
        config_path = config_dir / config
        if config_path.exists():
            print(f"✅ {config}")
        else:
            print(f"❌ {config} not found")
            return False
    
    return True

def test_cli_file():
    """Test that CLI file exists and is executable"""
    print("\n🧪 Testing CLI file...")
    
    cli_path = Path(__file__).parent / "cli.py"
    
    if not cli_path.exists():
        print("❌ cli.py not found")
        return False
    
    if not cli_path.is_file():
        print("❌ cli.py is not a file")
        return False
    
    # Check if executable (on Unix systems)
    if hasattr(cli_path, 'stat'):
        import stat
        if cli_path.stat().st_mode & stat.S_IEXEC:
            print("✅ cli.py is executable")
        else:
            print("⚠️  cli.py is not executable (run: chmod +x cli.py)")
    
    print("✅ cli.py exists")
    return True

def test_basic_syntax():
    """Test that CLI file has valid Python syntax"""
    print("\n🧪 Testing CLI syntax...")
    
    cli_path = Path(__file__).parent / "cli.py"
    
    try:
        with open(cli_path, 'r') as f:
            code = f.read()
        
        compile(code, str(cli_path), 'exec')
        print("✅ CLI syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in CLI: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading CLI: {e}")
        return False

def main():
    """Run all setup tests"""
    print("🚀 A2A ChatAgent CLI Setup Test\n")
    
    tests = [
        test_imports,
        test_config_files,
        test_cli_file,
        test_basic_syntax
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*50)
    
    if all(results):
        print("🎉 All tests passed! CLI is ready to use.")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements-cli.txt")
        print("2. Run CLI: python cli.py --interactive")
        print("3. Or run help: python cli.py --help")
        return 0
    else:
        failed_tests = len([r for r in results if not r])
        print(f"❌ {failed_tests} test(s) failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())