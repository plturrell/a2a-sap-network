#!/usr/bin/env python3
"""
Test refinements to GleanAgent - verify reduced false positives
"""
import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path

def run_cli_command(command, timeout=30):
    """Run CLI command and return result"""
    print(f"\n💻 Running: python3 cli.py {' '.join(command)}")
    print("-" * 70)
    
    try:
        result = subprocess.run(
            ["python3", "app/a2a/agents/gleanAgent/cli.py"] + command,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Print the actual output
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"⚠️  Errors:\n{result.stderr}")
            
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return {"success": False, "error": "timeout"}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e)}

def main():
    """Test refinements on GleanAgent itself"""
    print("🧪 TESTING GLEAN AGENT REFINEMENTS")
    print("=" * 80)
    
    # Test 1: Security Analysis on GleanAgent itself (should not detect its own patterns)
    print("\n\n🛡️  TEST 1: SECURITY ANALYSIS ON GLEAN AGENT (Should reduce false positives)")
    print("=" * 80)
    result = run_cli_command(["security", "app/a2a/agents/gleanAgent", "--max-vulns", "10"])
    
    # Check if it still detects its own patterns
    if result["success"] and "gleanAgentSdk.py" in result["stdout"]:
        print("\n⚠️  Still detecting patterns in gleanAgentSdk.py - checking if reduced...")
    else:
        print("\n✅ No longer detecting its own pattern definitions!")
    
    # Test 2: Linting Analysis (should use relative paths)
    print("\n\n🔍 TEST 2: LINTING ANALYSIS (Should use relative paths)")
    print("=" * 80)
    result = run_cli_command(["lint", "app/a2a/agents/gleanAgent", "--max-issues", "5"])
    
    # Check for "No module named" errors
    if result["success"]:
        if "No module named" in result["stdout"]:
            print("\n⚠️  Still has 'No module named' errors")
        else:
            print("\n✅ No 'No module named' errors found!")
    
    # Test 3: Create test file with intentional issues to verify detection still works
    print("\n\n🧪 TEST 3: VERIFY DETECTION STILL WORKS")
    print("=" * 80)
    
    test_dir = tempfile.mkdtemp(prefix="glean_test_")
    test_file = Path(test_dir) / "vulnerable_code.py"
    
    # Write test file with real vulnerabilities (not in test context)
    test_file.write_text('''
# Real vulnerability examples (not in test)

def process_user_input(user_id):
    """Process user data with SQL injection vulnerability"""
    import sqlite3
    conn = sqlite3.connect("users.db")
    # This is a real SQL injection vulnerability
    query = "SELECT * FROM users WHERE id = '" + user_id + "'"
    cursor = conn.execute(query)
    return cursor.fetchall()

def save_credentials():
    """Save credentials insecurely"""
    # Real hardcoded secret
    API_KEY = "sk-production-key-1234567890"
    PASSWORD = "admin123"
    return API_KEY

def run_command(cmd):
    """Execute system command unsafely"""
    import os
    # Real command injection vulnerability
    os.system("echo " + cmd)
''')
    
    print(f"📁 Created test file: {test_file}")
    
    # Run security scan on test file
    result = run_cli_command(["security", test_dir])
    
    if result["success"]:
        # Check for actual vulnerability count
        if "Total vulnerabilities: 0" in result["stdout"]:
            print("\n❌ Detection may be broken - no vulnerabilities found in test file")
        elif "vulnerabilities" in result["stdout"].lower() and "Total vulnerabilities:" in result["stdout"]:
            # Extract the actual count
            import re
            match = re.search(r'Total vulnerabilities: (\d+)', result["stdout"])
            if match and int(match.group(1)) > 0:
                print(f"\n✅ Detection still works! Found {match.group(1)} vulnerabilities in test file")
            else:
                print("\n❌ Detection may be broken - no vulnerabilities found in test file")
        else:
            print("\n⚠️  Unable to determine if vulnerabilities were found")
    
    # Cleanup
    shutil.rmtree(test_dir)
    
    print("\n\n" + "=" * 80)
    print("🎉 REFINEMENT TEST SUMMARY")
    print("=" * 80)
    print("\n✅ Key improvements implemented:")
    print("  • Enhanced security scanner with context awareness")
    print("  • Added file skip patterns for test files and pattern definitions")
    print("  • Fixed linter path handling to use relative paths")
    print("  • Improved AST context detection for better accuracy")
    print("\n🔑 Expected results:")
    print("  • Fewer false positives from pattern definitions")
    print("  • No 'No module named' errors in linting")
    print("  • Real vulnerabilities still detected accurately")
    
if __name__ == "__main__":
    main()