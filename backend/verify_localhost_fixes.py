#!/usr/bin/env python3
"""
Verification script to check localhost reference fixes and environment configuration.
Validates that critical files are properly configured for production deployment.
"""

import os
import re
import json
from typing import Dict, List, Tuple

def check_env_example() -> Tuple[bool, List[str]]:
    """Check if .env.example has proper environment variable fallbacks."""
    env_file = '/Users/apple/projects/a2a/a2aAgents/backend/.env.example'
    issues = []
    
    try:
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Check for required environment variables
        required_vars = [
            'A2A_RPC_URL', 'A2A_BASE_URL', 'DATA_MANAGER_URL', 
            'CATALOG_MANAGER_URL', 'AGENT_MANAGER_URL', 'CORS_ORIGINS',
            'PROMETHEUS_HOST', 'NODE_EXPORTER_HOST'
        ]
        
        for var in required_vars:
            if var not in content:
                issues.append(f"Missing environment variable: {var}")
        
        return len(issues) == 0, issues
        
    except Exception as e:
        return False, [f"Error reading .env.example: {e}"]

def check_docker_compose() -> Tuple[bool, List[str]]:
    """Check Docker Compose files for proper service names."""
    compose_file = '/Users/apple/projects/a2a/a2aAgents/backend/docker/docker-compose.infrastructure.yml'
    issues = []
    
    try:
        with open(compose_file, 'r') as f:
            content = f.read()
        
        # Check that health checks use service names, not localhost
        localhost_health_checks = re.findall(r'http://localhost:\d+', content)
        if localhost_health_checks:
            issues.append(f"Found {len(localhost_health_checks)} localhost references in health checks")
        
        return len(issues) == 0, issues
        
    except Exception as e:
        return False, [f"Error reading docker-compose.yml: {e}"]

def check_nginx_config() -> Tuple[bool, List[str]]:
    """Check nginx configuration for environment variable usage."""
    nginx_file = '/Users/apple/projects/a2a/a2aAgents/backend/nginx/nginx.conf'
    issues = []
    
    try:
        with open(nginx_file, 'r') as f:
            content = f.read()
        
        # Check for environment variable usage in server names
        if '${SERVER_NAME' not in content:
            issues.append("SERVER_NAME environment variable not used")
        
        if '${DOMAIN_NAME' not in content:
            issues.append("DOMAIN_NAME environment variable not used")
        
        return len(issues) == 0, issues
        
    except Exception as e:
        return False, [f"Error reading nginx.conf: {e}"]

def check_prometheus_config() -> Tuple[bool, List[str]]:
    """Check Prometheus configuration for environment variables."""
    prometheus_file = '/Users/apple/projects/a2a/a2aAgents/backend/monitoring/prometheus.yml'
    issues = []
    
    try:
        with open(prometheus_file, 'r') as f:
            content = f.read()
        
        # Check for environment variable usage
        if '${PROMETHEUS_HOST' not in content:
            issues.append("PROMETHEUS_HOST environment variable not used")
        
        if '${NODE_EXPORTER_HOST' not in content:
            issues.append("NODE_EXPORTER_HOST environment variable not used")
        
        return len(issues) == 0, issues
        
    except Exception as e:
        return False, [f"Error reading prometheus.yml: {e}"]

def check_main_py_cors() -> Tuple[bool, List[str]]:
    """Check main.py for proper CORS configuration."""
    main_file = '/Users/apple/projects/a2a/a2aAgents/backend/main.py'
    issues = []
    
    try:
        with open(main_file, 'r') as f:
            content = f.read()
        
        # Check for CORS_ORIGINS environment variable usage
        if 'CORS_ORIGINS' not in content:
            issues.append("CORS_ORIGINS environment variable not used in main.py")
        
        # Check that localhost is not hardcoded
        hardcoded_localhost = re.findall(r'"http://localhost:\d+"', content)
        if any('localhost' in line for line in hardcoded_localhost):
            issues.append("Found hardcoded localhost in CORS origins")
        
        return len(issues) == 0, issues
        
    except Exception as e:
        return False, [f"Error reading main.py: {e}"]

def check_agent_config() -> Tuple[bool, List[str]]:
    """Check agentConfig.py for proper environment configuration."""
    config_file = '/Users/apple/projects/a2a/a2aAgents/backend/config/agentConfig.py'
    issues = []
    
    try:
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Check for environment variable usage
        required_env_vars = ['A2A_BASE_URL', 'DATA_MANAGER_URL', 'CATALOG_MANAGER_URL']
        for var in required_env_vars:
            if f'os.getenv("{var}"' not in content:
                issues.append(f"Environment variable {var} not used in agentConfig.py")
        
        return len(issues) == 0, issues
        
    except Exception as e:
        return False, [f"Error reading agentConfig.py: {e}"]

def count_remaining_localhost_references() -> Dict[str, int]:
    """Count remaining localhost references in different file types."""
    base_path = '/Users/apple/projects/a2a/a2aAgents/backend'
    results = {
        'python_files': 0,
        'config_files': 0,
        'docker_files': 0,
        'total_files_with_localhost': 0
    }
    
    # Scan for localhost references
    for root, dirs, files in os.walk(base_path):
        # Skip test directories and node_modules
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'test', 'tests']]
        
        for file in files:
            if file.startswith('.') or file.endswith(('.pyc', '.pyo')):
                continue
                
            file_path = os.path.join(root, file)
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                
                if 'localhost' in content:
                    results['total_files_with_localhost'] += 1
                    
                    if file.endswith('.py'):
                        results['python_files'] += 1
                    elif file in ['nginx.conf', 'prometheus.yml', '.env.example']:
                        results['config_files'] += 1
                    elif file.startswith('docker-compose') or file == 'Dockerfile':
                        results['docker_files'] += 1
                        
            except (UnicodeDecodeError, PermissionError, IsADirectoryError):
                continue
    
    return results

def main():
    """Run comprehensive verification of localhost fixes."""
    print("🔍 Verifying localhost reference fixes and environment configuration...\n")
    
    # Check individual components
    checks = [
        ("Environment Configuration (.env.example)", check_env_example),
        ("Docker Compose Configuration", check_docker_compose),
        ("Nginx Configuration", check_nginx_config),
        ("Prometheus Configuration", check_prometheus_config),
        ("Main Application CORS", check_main_py_cors),
        ("Agent Configuration", check_agent_config),
    ]
    
    total_checks = len(checks)
    passed_checks = 0
    all_issues = []
    
    for check_name, check_func in checks:
        success, issues = check_func()
        
        if success:
            print(f"✅ {check_name}: PASSED")
            passed_checks += 1
        else:
            print(f"❌ {check_name}: FAILED")
            for issue in issues:
                print(f"   - {issue}")
            all_issues.extend(issues)
        
    print(f"\n📊 Configuration Check Results:")
    print(f"   ✅ Passed: {passed_checks}/{total_checks}")
    print(f"   ❌ Failed: {total_checks - passed_checks}/{total_checks}")
    
    # Count remaining localhost references
    print(f"\n🔍 Scanning for remaining localhost references...")
    localhost_count = count_remaining_localhost_references()
    
    print(f"📈 Localhost References Found:")
    print(f"   🐍 Python files: {localhost_count['python_files']}")
    print(f"   ⚙️  Config files: {localhost_count['config_files']}")
    print(f"   🐳 Docker files: {localhost_count['docker_files']}")
    print(f"   📁 Total files: {localhost_count['total_files_with_localhost']}")
    
    # Overall assessment
    print(f"\n🎯 Overall Assessment:")
    
    if passed_checks == total_checks and localhost_count['total_files_with_localhost'] < 50:
        print("✅ EXCELLENT: Configuration is production-ready")
        print("   All critical components properly use environment variables")
        print("   Minimal localhost references remaining (likely in tests/docs)")
    elif passed_checks >= total_checks * 0.8:
        print("🟡 GOOD: Configuration mostly ready for production")
        print("   Most critical components configured properly")
        print("   Some minor issues need attention before deployment")
    else:
        print("❌ NEEDS WORK: Configuration requires fixes before production")
        print("   Critical components still have hardcoded values")
        print("   Review and fix the issues listed above")
    
    # Recommendations
    print(f"\n📝 Recommendations:")
    if localhost_count['total_files_with_localhost'] > 0:
        print("   1. Review remaining localhost references for production impact")
        print("   2. Consider localhost in test files may be acceptable")
        print("   3. Validate all service URLs in production environment")
    
    if len(all_issues) > 0:
        print("   4. Address the specific issues identified above")
        print("   5. Test the application with production-like environment variables")
    
    print("   6. Document the required environment variables for deployment")
    print("   7. Create deployment scripts that validate environment configuration")
    
    print(f"\n🚀 Next Steps for Production Deployment:")
    print("   1. Set all required environment variables in production")
    print("   2. Test with actual service URLs (no localhost)")
    print("   3. Validate CORS origins for production domains")
    print("   4. Configure monitoring with actual Prometheus/Grafana URLs")
    print("   5. Update nginx server names for production domain")

if __name__ == "__main__":
    main()