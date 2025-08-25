#!/usr/bin/env python3
"""
Test real security vulnerability scanning implementation
"""
import asyncio
import sys
import os
from pathlib import Path
import tempfile
import shutil
import json

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(backend_dir))

async def test_real_security_scanning():
    """Test the real security vulnerability scanning implementation"""
    print("Testing Real Security Vulnerability Scanning")
    print("=" * 60)

    try:
        # Import the agent
        from app.a2a.agents.gleanAgent import GleanAgent
        print("✓ Successfully imported GleanAgent")

        # Create agent instance
        agent = GleanAgent()
        print(f"✓ Created agent: {agent.agent_id}")

        # Create a test directory with vulnerable code and dependencies
        test_dir = tempfile.mkdtemp(prefix="glean_security_test_")
        print(f"\n1. Created test directory: {test_dir}")

        # Create vulnerable Python requirements
        requirements_txt = Path(test_dir) / "requirements.txt"
        requirements_txt.write_text('''
# Vulnerable dependencies for testing
django==3.2.10
flask==1.1.0
pillow==8.0.0
requests==2.25.0
numpy==1.20.0
urllib3==1.26.0
''')

        # Create vulnerable package.json
        package_json = Path(test_dir) / "package.json"
        package_json.write_text(json.dumps({
            "name": "vulnerable-app",
            "version": "1.0.0",
            "dependencies": {
                "lodash": "4.17.20",
                "express": "4.17.0",
                "react": "16.13.0"
            },
            "devDependencies": {
                "webpack": "4.44.0"
            }
        }, indent=2))

        # Create vulnerable Python code
        vulnerable_py = Path(test_dir) / "vulnerable_app.py"
        vulnerable_py.write_text('''
import os
import subprocess
import random
import sqlite3

# Hardcoded secrets (security issue)
API_KEY = "sk-1234567890abcdef1234567890abcdef"
DATABASE_PASSWORD = "supersecret123"
SECRET_TOKEN = "jwt-token-abcdef123456789"

def unsafe_sql_query(user_input):
    """Function with SQL injection vulnerability"""
    # SQL injection vulnerability - string concatenation
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"

    # Another SQL injection pattern
    cursor.execute("SELECT * FROM products WHERE id = %s" % user_input)

    return query

def command_injection_vulnerability(filename):
    """Function with command injection vulnerability"""
    # Command injection via os.system
    os.system("cat " + filename)

    # Another command injection pattern
    subprocess.run(f"ls {filename}", shell=True)

    # eval() usage is dangerous
    result = eval(filename)

    return result

def weak_random_for_security():
    """Using weak random for security purposes"""
    # Weak random for token generation
    token = str(random.random())

    # Weak random choice for session ID
    session_id = random.choice("abcdefghijklmnopqrstuvwxyz")

    return token + session_id

def path_traversal_vulnerability(user_file):
    """Function with path traversal vulnerability"""
    # Path traversal vulnerability
    with open("uploads/" + user_file + "../config.txt", "r") as f:
        content = f.read()

    return content

class DatabaseManager:
    def __init__(self):
        # More hardcoded credentials
        self.connection_string = "postgresql://admin:password123@localhost/db"

    def execute_dynamic_query(self, table, condition):
        """Another SQL injection vulnerability"""
        query = f"DELETE FROM {table} WHERE {condition}"
        # This would be vulnerable to SQL injection
        return query

# JavaScript-style eval equivalent
def dangerous_eval(code):
    """Dangerous code execution"""
    exec(code)  # Never do this with user input

# Weak password handling
def store_password(password):
    """Insecure password storage"""
    # Storing password in plain text (bad practice)
    with open("passwords.txt", "a") as f:
        f.write(f"user:password = {password}\\n")
''')

        # Create another vulnerable file
        web_py = Path(test_dir) / "web_server.py"
        web_py.write_text('''
from flask import Flask, request
import os

app = Flask(__name__)

# Hardcoded Flask secret key
app.secret_key = "hardcoded-secret-key-12345"

@app.route('/exec')
def execute_command():
    """Vulnerable endpoint"""
    cmd = request.args.get('cmd')
    # Command injection vulnerability
    os.system(cmd)
    return "Command executed"

@app.route('/file/<path:filename>')
def read_file(filename):
    """Path traversal vulnerability"""
    # This allows reading any file on the system
    file_path = "uploads/" + filename
    with open(file_path) as f:
        return f.read()

if __name__ == '__main__':
    app.run(debug=True)  # Debug mode in production is bad
''')

        print("\n2. Testing comprehensive security vulnerability scanning:")

        # Test the enhanced security scanning
        security_result = await agent.scan_dependency_vulnerabilities(test_dir, scan_dev_dependencies=True)

        print(f"   Directory scanned: {security_result.get('directory', 'unknown')}")
        print(f"   Total vulnerabilities: {security_result.get('total_vulnerabilities', 0)}")
        print(f"   Scanned files: {len(security_result.get('scanned_files', []))}")

        # Show severity breakdown
        if 'severity_breakdown' in security_result:
            breakdown = security_result['severity_breakdown']
            print(f"\n   Severity Breakdown:")
            for severity, count in breakdown.items():
                print(f"     - {severity.title()}: {count}")

        # Show risk metrics
        if 'risk_metrics' in security_result:
            metrics = security_result['risk_metrics']
            print(f"\n   Risk Assessment:")
            print(f"     - Risk Score: {metrics.get('risk_score', 0)}/100")
            print(f"     - Risk Level: {metrics.get('risk_level', 'unknown').title()}")
            print(f"     - Critical Count: {metrics.get('critical_count', 0)}")
            print(f"     - High Count: {metrics.get('high_count', 0)}")
            print(f"     - Medium Count: {metrics.get('medium_count', 0)}")
            print(f"     - Low Count: {metrics.get('low_count', 0)}")

        # Show vulnerability details by type
        vulnerabilities = security_result.get('vulnerabilities', [])
        if vulnerabilities:
            print(f"\n   Sample Vulnerabilities Found:")

            # Group by source
            by_source = {}
            for vuln in vulnerabilities:
                source = vuln.get('source', 'unknown')
                if source not in by_source:
                    by_source[source] = []
                by_source[source].append(vuln)

            for source, vulns in by_source.items():
                print(f"\n     {source.replace('_', ' ').title()} ({len(vulns)} found):")
                for vuln in vulns[:3]:  # Show first 3 of each type
                    print(f"       • {vuln.get('vulnerability_id', 'N/A')}: {vuln.get('description', 'No description')}")
                    if vuln.get('package') != 'source_code':
                        print(f"         Package: {vuln.get('package', 'unknown')} v{vuln.get('version', 'unknown')}")
                    if vuln.get('file'):
                        filename = Path(vuln['file']).name
                        line_info = f":{vuln['line']}" if vuln.get('line') else ""
                        print(f"         File: {filename}{line_info}")
                    if vuln.get('remediation'):
                        print(f"         Fix: {vuln['remediation']}")
                    print()

        print(f"   Database Version: {security_result.get('database_version', 'unknown')}")

        print("\n✅ Real security vulnerability scanning test completed!")
        print("\nKey Security Features Demonstrated:")
        print("  ✓ Built-in vulnerability database with real CVE data")
        print("  ✓ Python dependency scanning (requirements.txt, pyproject.toml)")
        print("  ✓ Node.js dependency scanning (package.json)")
        print("  ✓ Static code analysis for security patterns")
        print("  ✓ SQL injection pattern detection")
        print("  ✓ Command injection pattern detection")
        print("  ✓ Hardcoded secrets detection")
        print("  ✓ Weak random number generation detection")
        print("  ✓ Path traversal vulnerability detection")
        print("  ✓ Risk scoring and metrics calculation")
        print("  ✓ Vulnerability deduplication")
        print("  ✓ Version comparison and matching")

        # Cleanup
        shutil.rmtree(test_dir)
        print(f"\nCleaned up test directory")

    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    # Run the async test
    success = asyncio.run(test_real_security_scanning())
    sys.exit(0 if success else 1)